import pytest
import torch

from helpers import MODEL_NAMES, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP

EXPECTED_LOSSES = {
    "fusion_concat": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_multilearner": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_none": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "image_mae": {"loss", "loss_mae"},
    "text_mlm": {"loss", "loss_mlm"},
}


def tiny_model(name: str, *overrides: str) -> MultiMAE:
    torch.manual_seed(0)
    cfg = compose_cfg(f"model={name}", f"model.backbone.pretrained={TINY_CLIP}", *overrides)
    return MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_returns_expected_finite_losses(name, tokenizer):
    out = tiny_model(name)(make_batch(tokenizer))
    assert set(out) == EXPECTED_LOSSES[name]
    assert all(torch.isfinite(v) and v.ndim == 0 for v in out.values())
    parts = sum(v for k, v in out.items() if k != "loss")
    assert torch.allclose(out["loss"], parts)  # all weights are 1.0 by default


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_trainable_parameter_gets_a_gradient(name, tokenizer):
    model = tiny_model(name)
    model(make_batch(tokenizer))["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing  # DDP (find_unused_parameters=False) needs this


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_frozen_backbones_get_no_gradient(name, tokenizer):
    model = tiny_model(name, "model.freeze_backbones=true")
    model(make_batch(tokenizer))["loss"].backward()
    for tower in (model.vision, model.text):
        if tower is not None:
            assert all(not p.requires_grad and p.grad is None for p in tower.pretrained_parameters())
    if model.text is not None:
        assert model.text.mask_embedding.grad is not None
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_mean_pooling_trains_the_new_projection(tokenizer):
    model = tiny_model("fusion_concat", "model.pooling=mean")
    model(make_batch(tokenizer))["loss"].backward()
    assert model.vision.mean_projection.weight.grad is not None
    assert not model.vision.projection.weight.requires_grad  # native head unused, frozen for DDP
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_tiny_batch_overfits(name, tokenizer):
    model = tiny_model(name)
    batch = make_batch(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(150):
        out = model(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        history.append({k: v.item() for k, v in out.items()})
    for key in ("loss_mae", "loss_mlm"):
        if key in history[0]:
            first = sum(h[key] for h in history[:5]) / 5
            last = sum(h[key] for h in history[-5:]) / 5
            assert last < 0.7 * first, (key, first, last)


def test_empty_caption_in_batch_is_finite(tokenizer):
    batch = make_batch(tokenizer, captions=["", "a dog", "", "two cats on a bed"])
    out = tiny_model("fusion_concat")(batch)
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()


def test_bf16_autocast_losses_are_finite_float32(tokenizer):
    model = tiny_model("fusion_multilearner")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(make_batch(tokenizer))
    assert all(v.dtype == torch.float32 and torch.isfinite(v) for v in out.values())


def test_embeddings_are_normalized(tokenizer):
    model = tiny_model("fusion_concat").eval()
    batch = make_batch(tokenizer)
    with torch.no_grad():
        img = model.embed_image(batch["pixel_values"])
        txt = model.embed_text(batch["input_ids"], batch["attention_mask"])
    assert torch.allclose(img.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(txt.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_param_groups_partition_trainable_parameters():
    model = tiny_model("fusion_concat")
    groups = model.param_groups(lr=1e-4, lr_backbone=1e-5, weight_decay=0.05)
    seen = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable) and len(seen) == len(set(seen))
    by_name = {g["name"]: g for g in groups}
    assert by_name["backbone_decay"]["lr"] == 1e-5 and by_name["head_decay"]["lr"] == 1e-4
    no_decay = {id(p) for g in groups if g["weight_decay"] == 0.0 for p in g["params"]}
    for param in (
        model.logit_scale,
        model.text.mask_embedding,
        model.image_decoder.queries,
        model.image_decoder.pos_embed,
        model.vision.model.embeddings.position_embedding.weight,
        model.text.model.embeddings.token_embedding.weight,
        model.fusion.type_embed,
    ):
        assert id(param) in no_decay
    assert id(model.image_proj.weight) not in no_decay


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        tiny_model("fusion_concat", "model.modalities=[audio]")
    with pytest.raises(ValueError):
        tiny_model("fusion_concat", "data.max_text_len=100")  # CLIP has 77 positions
