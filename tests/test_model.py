import pytest
import torch

from helpers import MODEL_NAMES, compose_cfg, make_batch
from mmae.losses import mae_loss, mlm_loss
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP

STEPS = 300

EXPECTED_LOSSES = {
    "fusion_concat": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_multilearner": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_none": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "contrastive": {"loss", "loss_contrastive"},
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


@pytest.mark.parametrize("pooling", ["native", "mean"])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_trainable_parameter_gets_a_gradient(name, pooling, tokenizer):
    model = tiny_model(name, f"model.pooling={pooling}")
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
    if model.text is not None and model.reconstruction:
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


# Four captions with the same token count, so padding cannot identify a row.
EQUAL_LENGTH_CAPTIONS = ["a dog on the beach", "two men on big horses", "a cat on a bed", "a red bus in town"]


@pytest.mark.slow
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_tiny_batch_overfits(name, tokenizer):
    """Learning smoke check: the reconstruction losses fall on a fixed tiny batch.

    It does not prove the decoders use the visible inputs; test_decoders_see_only_visible_inputs
    is the masking guard.
    """
    model = tiny_model(name)
    batch = make_batch(tokenizer, captions=EQUAL_LENGTH_CAPTIONS)
    assert len(set(batch["attention_mask"].sum(dim=1).tolist())) == 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(STEPS):
        out = model(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        history.append({k: v.item() for k, v in out.items()})
    for key in ("loss_mae", "loss_mlm"):
        if key in history[0]:
            first = sum(h[key] for h in history[:5]) / 5
            last = sum(h[key] for h in history[-5:]) / 5
            print(name, key, round(first, 3), round(last, 3))
            assert last < 0.7 * first, (key, first, last)


def test_multilearner_needs_both_modalities():
    for name in ("image_mae", "text_mlm"):
        with pytest.raises(ValueError, match="both modalities"):
            tiny_model(name, "model.fusion.type=multilearner")


def recorder(model: MultiMAE, monkeypatch):
    """Record the masks forward draws and capture the fusion output, the decoder outputs and the losses.

    Masks are recorded by wrapping random_patch_mask / random_token_mask in the model module. run(batch)
    seeds the RNG before the forward pass, so every call draws the same masks. Returns (masks, run), where
    run(batch) -> (decoder outputs by decoder name, losses, fusion output).
    """
    import mmae.models.model as model_module

    masks = {}
    for attr, key in (("random_patch_mask", "patch"), ("random_token_mask", "token")):
        original = getattr(model_module, attr)

        def wrapped(*args, _original=original, _key=key, **kwargs):
            result = _original(*args, **kwargs)
            masks[_key] = result[1] if _key == "patch" else result
            return result

        monkeypatch.setattr(model_module, attr, wrapped)
    captured = {}
    for dec in ("image_decoder", "text_decoder"):
        if hasattr(model, dec):
            getattr(model, dec).register_forward_hook(lambda m, i, o, _k=dec: captured.__setitem__(_k, o.detach().clone()))
    model.fusion.register_forward_hook(lambda m, i, o: captured.__setitem__("fusion", o))

    def run(batch):
        torch.manual_seed(123)
        captured.clear()
        with torch.no_grad():
            losses = model(batch)
        fused = captured.pop("fusion")
        return dict(captured), losses, fused

    return masks, run


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "contrastive"])  # no masked pass
def test_decoders_see_only_visible_inputs(name, tokenizer, monkeypatch):
    """Changing masked content leaves the decoder outputs bit-identical; changing visible content moves them.
    The MAE and MLM losses are exactly mae_loss / mlm_loss of the decoder outputs over the recorded masks."""
    model = tiny_model(name).eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base, losses, _ = run(batch)
    patch_mask, token_mask = masks.get("patch"), masks.get("token")
    again = run(batch)[0]
    assert base and again.keys() == base.keys() and all(torch.equal(base[k], again[k]) for k in base)

    # the losses score exactly the masked positions of the decoder outputs
    if model.use_image:
        expected = mae_loss(base["image_decoder"], batch["pixel_values"], patch_mask, model.vision.patch_size, model.norm_pix)
        assert torch.equal(losses["loss_mae"], expected), (losses["loss_mae"], expected)
    if model.use_text:
        expected = mlm_loss(base["text_decoder"], batch["input_ids"], token_mask)
        assert torch.equal(losses["loss_mlm"], expected), (losses["loss_mlm"], expected)

    ps = model.vision.patch_size if model.use_image else None
    grid = 224 // ps if ps else None
    noise = torch.randn_like(batch["pixel_values"]) * 50
    random_ids = torch.randint(1000, 40000, batch["input_ids"].shape)

    def pixel_region(mask):  # patch order is row-major, as in patchify
        m = mask.view(-1, 1, grid, grid).float()
        return m.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3).bool()

    def changed(masked_part: bool):
        new = dict(batch)
        if model.use_image:
            region = pixel_region(patch_mask) if masked_part else ~pixel_region(patch_mask)
            new["pixel_values"] = torch.where(region, noise, batch["pixel_values"])
        if model.use_text:
            region = token_mask if masked_part else (
                batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool() & ~token_mask
            )
            new["input_ids"] = torch.where(region, random_ids, batch["input_ids"])
        return new

    masked_changed = run(changed(True))[0]
    visible_changed = run(changed(False))[0]
    for key in base:
        assert torch.equal(base[key], masked_changed[key]), f"{key} depends on masked content"
        assert not torch.equal(base[key], visible_changed[key]), f"{key} ignores visible content"


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n not in ("image_mae", "contrastive")])
def test_text_padding_reaches_no_real_position(name, tokenizer, monkeypatch):
    """Padded text positions reach neither decoder at real positions nor any loss.

    1. Garbage token ids at padded positions (attention_mask unchanged) enter the fused memory there, yet
       every decoder output at a real position and every loss stay bit-identical: the fusion and both
       decoders get the text padding as a memory mask.
    2. Perturbing the text decoder's queries with random noise at positions padded in every row moves its
       outputs there by a clear margin yet leaves its outputs at real positions bit-identical: the decoder
       gets the padding as a query (self-attention) mask too.
    """
    model = tiny_model(name).eval()
    _, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    real = batch["attention_mask"].bool()
    assert (~real).any(dim=1).all()  # every caption is padded
    base, base_losses, base_fused = run(batch)

    garbage = dict(batch, input_ids=torch.where(real, batch["input_ids"], torch.randint(1000, 40000, real.shape)))
    out, losses, fused = run(garbage)
    pad_memory = torch.cat([torch.zeros_like(base_fused.text_padding[:, : -real.shape[1]]), ~real], dim=1)
    assert torch.equal(base_fused.text_padding, pad_memory)
    assert not torch.equal(base_fused.text_memory[pad_memory], fused.text_memory[pad_memory])  # garbage got in
    assert torch.equal(out["text_decoder"][real], base["text_decoder"][real])
    if "image_decoder" in base:
        assert torch.equal(out["image_decoder"], base["image_decoder"])
    assert losses.keys() == base_losses.keys()
    for key in base_losses:
        assert torch.equal(losses[key], base_losses[key]), key

    all_padded = ~real.any(dim=0)
    assert all_padded.sum() >= 8
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        noise = torch.randn(int(all_padded.sum()), model.text_decoder.queries.shape[1], generator=g)
        model.text_decoder.queries[all_padded] += noise
    out, losses, _ = run(batch)
    moved = (out["text_decoder"][:, all_padded] - base["text_decoder"][:, all_padded]).abs().max()
    assert moved > 1e-2, moved  # the perturbation really reaches the padded queries
    assert torch.equal(out["text_decoder"][real], base["text_decoder"][real])
    assert torch.equal(losses["loss_mlm"], base_losses["loss_mlm"])


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


def test_tower_layer_id():
    from mmae.models.model import tower_layer_id

    assert tower_layer_id("model.embeddings.patch_embedding.weight", 12) == 0
    assert tower_layer_id("model.pre_layrnorm.weight", 12) == 0
    assert tower_layer_id("model.encoder.layers.0.mlp.fc1.weight", 12) == 1
    assert tower_layer_id("model.encoder.layers.11.mlp.fc1.weight", 12) == 12
    assert tower_layer_id("model.post_layernorm.weight", 12) == 13
    assert tower_layer_id("projection.weight", 12) == 13


def test_split_param_groups_per_tower_and_layer():
    model = tiny_model("fusion_concat")
    groups = model.param_groups(1e-4, 1e-5, 0.05, lr_text=5e-5, lr_vision=5e-6, layer_decay=0.5)
    seen = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable) and len(seen) == len(set(seen))

    def lr_of(param):
        return next(g["lr"] for g in groups if any(p is param for p in g["params"]))

    layers = len(model.vision.model.encoder.layers)  # 2 in the tiny CLIP
    assert lr_of(model.text.projection.weight) == pytest.approx(5e-5)                    # top: no decay
    assert lr_of(model.vision.projection.weight) == pytest.approx(5e-6)
    assert lr_of(model.vision.model.encoder.layers[0].mlp.fc1.weight) == pytest.approx(5e-6 * 0.5 ** layers)
    assert lr_of(model.vision.model.embeddings.patch_embedding.weight) == pytest.approx(5e-6 * 0.5 ** (layers + 1))
    assert lr_of(model.image_proj.weight) == pytest.approx(1e-4)                          # new modules
    assert lr_of(model.text.mask_embedding) == pytest.approx(1e-4)
    towers = {g["tower"] for g in groups if any(p is model.vision.projection.weight for p in g["params"])}
    assert towers == {"vision"}
    assert all(g["name"].startswith("backbone") == (g["tower"] is not None) for g in groups)
