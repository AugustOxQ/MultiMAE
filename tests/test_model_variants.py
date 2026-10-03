"""Default-off model switches of the multilearner line (spec 2026-10-03, section 8): M1, M2b, M3, M6."""
import pytest
import torch
from omegaconf import OmegaConf

from helpers import add_content_mask, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP
from test_model import recorder, tiny_model

# name -> overrides; Tasks 3 to 5 add their switches here.
VARIANTS = {
    "m1_clean": ("model.mlm_image_source=clean",),
    "m1_clean_detached": ("model.mlm_image_source=clean_detached",),
    "m2b_content": ("model.masking.text_mode=content",),
}


def variant_batch(tokenizer, variant: str) -> dict:
    if variant == "m2b_content":
        return add_content_mask(make_batch(tokenizer), tokenizer)
    return make_batch(tokenizer)


def change(batch: dict, model: MultiMAE, masks: dict, modality: str, masked_part: bool) -> dict:
    """Replace one modality's masked (or visible) content with noise; the other modality is left alone."""
    new = dict(batch)
    if modality == "image":
        ps = model.vision.patch_size
        grid = 224 // ps
        region = masks["patch"].view(-1, 1, grid, grid).float()
        region = region.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3).bool()
        region = region if masked_part else ~region
        new["pixel_values"] = torch.where(region, torch.randn_like(batch["pixel_values"]) * 50, batch["pixel_values"])
    else:
        token_mask = masks["token"]
        visible = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool() & ~token_mask
        region = token_mask if masked_part else visible
        new["input_ids"] = torch.where(region, torch.randint(1000, 40000, batch["input_ids"].shape), batch["input_ids"])
    return new


@pytest.mark.parametrize("pooling", ["native", "mean"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_gives_every_trainable_parameter_a_gradient(variant, pooling, tokenizer):
    model = tiny_model("fusion_multilearner", f"model.pooling={pooling}", *VARIANTS[variant])
    out = model(variant_batch(tokenizer, variant))
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_decoder_ignores_its_own_masked_content(variant, tokenizer, monkeypatch):
    """No switch may leak a modality's masked content into the decoder that reconstructs it."""
    model = tiny_model("fusion_multilearner", *VARIANTS[variant]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = variant_batch(tokenizer, variant)
    base = run(batch)[0]
    assert torch.equal(run(change(batch, model, masks, "image", True))[0]["image_decoder"], base["image_decoder"])
    assert torch.equal(run(change(batch, model, masks, "text", True))[0]["text_decoder"], base["text_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "image", False))[0]["image_decoder"], base["image_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "text", False))[0]["text_decoder"], base["text_decoder"])


def test_model_builds_from_a_config_without_the_new_keys():
    """Run configs saved before these switches existed (the 12 baseline runs) still build, with the defaults."""
    cfg = compose_cfg("model=fusion_multilearner", f"model.backbone.pretrained={TINY_CLIP}")
    old = OmegaConf.to_container(cfg.model)
    for key in ("mlm_image_source", "pooled_conditioning"):
        old.pop(key, None)
    old["masking"].pop("text_mode", None)
    old["loss"]["weights"].pop("masked_view", None)
    model = MultiMAE(OmegaConf.create(old), max_text_len=cfg.data.max_text_len)
    assert model.mlm_image_source == "masked"
    assert model.text_mode == "random"


@pytest.mark.parametrize("source", ["clean", "clean_detached"])
def test_m1_text_decoder_reads_the_masked_patches(source, tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", f"model.mlm_image_source={source}").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[0]
    moved = run(change(batch, model, masks, "image", True))[0]
    assert not torch.equal(moved["text_decoder"], base["text_decoder"])


@pytest.mark.parametrize(("source", "reaches_vision"), [("clean", True), ("clean_detached", False)])
def test_m1_detached_stops_the_mlm_gradient_into_the_vision_tower(source, reaches_vision, tokenizer):
    model = tiny_model("fusion_multilearner", f"model.mlm_image_source={source}")
    model(make_batch(tokenizer))["loss_mlm"].backward()
    grad = model.vision.model.encoder.layers[0].mlp.fc1.weight.grad
    assert (grad is not None and grad.abs().sum() > 0) == reaches_vision


@pytest.mark.parametrize("name", ["fusion_none", "contrastive"])
def test_m1_needs_a_fusion_that_reaches_the_text_decoder(name):
    with pytest.raises(ValueError, match="mlm_image_source"):
        tiny_model(name, "model.mlm_image_source=clean")


def test_m1_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="mlm_image_source"):
        tiny_model("fusion_multilearner", "model.mlm_image_source=sideways")


def test_m2b_masks_only_content_words(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content").eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_content_mask(make_batch(tokenizer), tokenizer)
    run(batch)
    assert masks["token"].any()
    assert not (masks["token"] & ~batch["content_tokens_mask"]).any()


def test_m2b_caption_without_content_words_is_finite(tokenizer):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content")
    batch = add_content_mask(make_batch(tokenizer, captions=["the of and", "", "a dog", "on the"]), tokenizer)
    out = model(batch)
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()


def test_m2b_without_the_content_mask_fails_clearly(tokenizer):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content")
    with pytest.raises(KeyError, match="content_tokens_mask"):
        model(make_batch(tokenizer))
