import pytest
import torch
import torch.nn.functional as F

from helpers import run_ddp_worker
from mmae.losses import MAX_LOGIT_SCALE, contrastive_loss, mae_loss, mlm_loss, patchify


def normalized(n: int, d: int, seed: int) -> torch.Tensor:
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def reference_infonce(image, text, scale):
    logits = scale * image @ text.T
    labels = torch.arange(image.shape[0])
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def test_contrastive_matches_reference():
    image, text = normalized(8, 6, 0), normalized(8, 6, 1)
    loss = contrastive_loss(image, text, torch.tensor(2.0))
    assert torch.allclose(loss, reference_infonce(image, text, torch.tensor(2.0).exp()), atol=1e-6)


def test_contrastive_does_not_clamp_logit_scale():
    """The loss uses exp(logit_scale) as is; the trainer clamps the parameter after each optimizer step
    (test_trainer.py::test_logit_scale_is_clamped_after_each_step)."""
    image, text = normalized(8, 6, 0), normalized(8, 6, 1)
    assert torch.tensor(5.0).exp() > MAX_LOGIT_SCALE
    loss = contrastive_loss(image, text, torch.tensor(5.0))
    assert torch.allclose(loss, reference_infonce(image, text, torch.tensor(5.0).exp()), atol=1e-5)


CLIP_LOGIT_SCALE = 4.605170249938965  # openai/clip-vit-base-patch32's pretrained value (float32 of log 100)


def test_logit_scale_gets_a_gradient_at_clips_pretrained_value():
    """exp(4.605170249938965) is 100.0000076 in float32, so a clamp of exp() at 100 inside the loss would
    give the parameter a zero gradient from the first step."""
    image, text = normalized(8, 6, 0), normalized(8, 6, 1)
    logit_scale = torch.tensor(CLIP_LOGIT_SCALE, requires_grad=True)
    assert logit_scale.exp() > MAX_LOGIT_SCALE
    contrastive_loss(image, text, logit_scale).backward()
    assert logit_scale.grad is not None and logit_scale.grad.abs() > 1e-3, logit_scale.grad


def test_contrastive_gather_matches_single_process(tmp_path):
    result = run_ddp_worker("contrastive", tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [torch.load(tmp_path / f"rank{r}.pt") for r in range(2)]
    # same draws as the worker: one generator, image first then text
    g = torch.Generator().manual_seed(0)
    image = F.normalize(torch.randn(8, 6, generator=g), dim=-1).requires_grad_(True)
    text = F.normalize(torch.randn(8, 6, generator=g), dim=-1).requires_grad_(True)
    single = contrastive_loss(image, text, torch.tensor(2.0), gather=False)
    single.backward()
    # the mean of the per-rank losses is the single-process loss
    assert torch.allclose((ranks[0]["loss"] + ranks[1]["loss"]) / 2, single.detach(), atol=1e-6)
    # each rank's gradient (summed over ranks by the gather) is world_size x the single-process gradient
    for r in range(2):
        rows = slice(r * 4, (r + 1) * 4)
        assert torch.allclose(ranks[r]["img_grad"], 2 * image.grad[rows], atol=1e-6)
        assert torch.allclose(ranks[r]["txt_grad"], 2 * text.grad[rows], atol=1e-6)


def test_patchify_matches_vit_patch_order():
    images = torch.randn(2, 3, 64, 64)
    patches = patchify(images, 32)
    assert patches.shape == (2, 4, 32 * 32 * 3)
    for row in range(2):
        for col in range(2):
            expected = images[1, :, row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32].permute(1, 2, 0).reshape(-1)
            assert torch.equal(patches[1, row * 2 + col], expected)


@pytest.mark.parametrize("norm_pix", [True, False])
def test_mae_loss_counts_masked_patches_only(norm_pix):
    images = torch.randn(2, 3, 64, 64)
    mask = torch.tensor([[True, False, True, False], [False, False, False, True]])
    pred = torch.randn(2, 4, 32 * 32 * 3)
    loss = mae_loss(pred, images, mask, 32, norm_pix=norm_pix)
    changed = pred.clone()
    changed[~mask] += 5.0  # visible patches must not matter
    assert torch.allclose(mae_loss(changed, images, mask, 32, norm_pix=norm_pix), loss)
    target = patchify(images, 32)
    if norm_pix:
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
    manual = ((pred - target) ** 2).mean(-1)[mask].mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_mlm_loss_counts_masked_tokens_only():
    logits = torch.randn(2, 5, 11)
    ids = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[False, True, False, False, False], [False, False, True, True, False]])
    loss = mlm_loss(logits, ids, mask)
    assert torch.allclose(loss, F.cross_entropy(logits[mask], ids[mask]))
    changed = logits.clone()
    changed[~mask] += 3.0
    assert torch.allclose(mlm_loss(changed, ids, mask), loss)


def test_mlm_loss_with_no_masked_tokens_is_zero_with_grad():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    loss = mlm_loss(logits, torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.bool))
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_losses_return_float32_for_bf16_inputs():
    image, text = normalized(4, 6, 0).bfloat16(), normalized(4, 6, 1).bfloat16()
    assert contrastive_loss(image, text, torch.tensor(2.0)).dtype == torch.float32
    pred = torch.randn(2, 4, 3072).bfloat16()
    assert mae_loss(pred, torch.randn(2, 3, 64, 64), torch.ones(2, 4, dtype=torch.bool), 32).dtype == torch.float32
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert mlm_loss(torch.randn(2, 5, 11).bfloat16(), torch.zeros(2, 5, dtype=torch.long), mask).dtype == torch.float32


def test_losses_are_fp32_under_autocast():
    image, text = normalized(16, 32, 0), normalized(16, 32, 1)
    scale = torch.tensor(100.0).log()
    plain = contrastive_loss(image, text, scale)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = contrastive_loss(image, text, scale)
    assert auto.dtype == torch.float32
    assert torch.allclose(auto, plain, atol=1e-6), (auto.item(), plain.item())

    images, pred = torch.randn(2, 3, 64, 64), torch.randn(2, 4, 3072)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    plain = mae_loss(pred, images, mask, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = mae_loss(pred, images, mask, 32)
    assert torch.allclose(auto, plain, atol=1e-6), (auto.item(), plain.item())

    logits, ids = torch.randn(2, 5, 11), torch.randint(0, 11, (2, 5))
    tmask = torch.tensor([[False, True, False, True, False], [True, False, False, True, False]])
    plain = mlm_loss(logits, ids, tmask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = mlm_loss(logits, ids, tmask)
    assert torch.allclose(auto, plain, atol=1e-6), (auto.item(), plain.item())
