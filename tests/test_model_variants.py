"""Default-off model switches of the multilearner line (spec 2026-10-03, section 8): M1, M2b, M3, M6."""
import pytest
import torch
from omegaconf import OmegaConf

from helpers import add_content_mask, add_emotion, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP
from test_model import recorder, tiny_model

# name -> overrides; Tasks 3 to 5 add their switches here.
VARIANTS = {
    "m1_clean": ("model.mlm_image_source=clean",),
    "m1_clean_detached": ("model.mlm_image_source=clean_detached",),
    "m2b_content": ("model.masking.text_mode=content",),
    "m3_pooled": ("model.pooled_conditioning=true",),
    "m6_masked_view": ("model.loss.weights.masked_view=0.25",),
    "emotion_head": ("model.emotion_head=true",),
    "emotion_ml80": ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0"),
    "parcap": ("model.emotion_head=true", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean", "model.loss.weights.mae=0"),
}

NO_VISIBLE_TEXT = {"parcap"}
EMOTION_VARIANTS = {"emotion_head", "emotion_ml80", "parcap"}


def variant_batch(tokenizer, variant: str) -> dict:
    batch = add_content_mask(make_batch(tokenizer), tokenizer) if variant == "m2b_content" else make_batch(tokenizer)
    return add_emotion(batch) if variant in EMOTION_VARIANTS else batch


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
    if variant not in NO_VISIBLE_TEXT:
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
    assert model.pooled_conditioning is False
    assert model.masked_view_weight == 0.0


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


def test_m3_each_decoder_reads_the_other_modality(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", "model.pooled_conditioning=true").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[0]
    assert not torch.equal(run(change(batch, model, masks, "image", True))[0]["text_decoder"], base["text_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "text", True))[0]["image_decoder"], base["image_decoder"])


@pytest.mark.parametrize("name", ["fusion_none", "fusion_concat"])
def test_m3_works_with_other_fusions(name, tokenizer):
    model = tiny_model(name, "model.pooled_conditioning=true")
    model(make_batch(tokenizer))["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", ["contrastive", "image_mae", "text_mlm"])
def test_m3_needs_both_modalities_and_reconstruction(name):
    with pytest.raises(ValueError, match="pooled_conditioning"):
        tiny_model(name, "model.pooled_conditioning=true")


def test_m6_adds_a_weighted_masked_view_loss(tokenizer):
    model = tiny_model("fusion_multilearner", "model.loss.weights.masked_view=0.25")
    out = model(make_batch(tokenizer))
    assert set(out) == {"loss", "loss_contrastive", "loss_mae", "loss_mlm", "loss_masked_view"}
    expected = out["loss_contrastive"] + out["loss_mae"] + out["loss_mlm"] + 0.25 * out["loss_masked_view"]
    assert torch.allclose(out["loss"], expected)
    assert "loss_masked_view" not in tiny_model("fusion_multilearner")(make_batch(tokenizer))


def test_m6_scores_the_masked_caption(tokenizer, monkeypatch):
    """The loss uses the masked view: masked tokens' content does not move it, visible tokens' content does."""
    model = tiny_model("fusion_multilearner", "model.loss.weights.masked_view=0.25").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[1]["loss_masked_view"]
    assert torch.equal(run(change(batch, model, masks, "text", True))[1]["loss_masked_view"], base)
    assert not torch.equal(run(change(batch, model, masks, "text", False))[1]["loss_masked_view"], base)


@pytest.mark.parametrize("name", ["contrastive", "image_mae", "text_mlm"])
def test_m6_needs_both_modalities_and_reconstruction(name):
    with pytest.raises(ValueError, match="masked_view"):
        tiny_model(name, "model.loss.weights.masked_view=0.25")


def test_parcap_text_decoder_sees_no_caption_content(tokenizer, monkeypatch):
    """At text ratio 1.0 the caption decoder's output depends on the image and the caption length only."""
    model = tiny_model("fusion_multilearner", *VARIANTS["parcap"]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_emotion(make_batch(tokenizer))
    base = run(batch)[0]["text_decoder"]
    real = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    assert torch.equal(masks["token"], real)
    other = dict(batch, input_ids=torch.where(real, torch.randint(1000, 40000, batch["input_ids"].shape), batch["input_ids"]))
    assert torch.equal(run(other)[0]["text_decoder"], base)
    noisy = dict(batch, pixel_values=batch["pixel_values"] + torch.randn_like(batch["pixel_values"]))
    assert not torch.equal(run(noisy)[0]["text_decoder"], base)
