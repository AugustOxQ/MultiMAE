import pytest
import torch
import torch.nn.functional as F

from helpers import CLIP_NAME
from mmae.models.backbones import (
    TINY_CLIP,
    ClipTextTower,
    ClipVisionTower,
    build_backbone,
    load_clip,
)

CAPTIONS = ["a photo of a cat on a mat", "dog", "two people riding horses on the beach today"]


def towers_for(name: str):
    torch.manual_seed(0)
    clip = load_clip(name).eval()
    return clip, ClipVisionTower(clip).eval(), ClipTextTower(clip).eval()


def text_inputs(tokenizer, max_len: int = 16):
    enc = tokenizer(CAPTIONS, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def unpadded_text_inputs(tokenizer):
    """Batches with an all-ones attention mask: HF then skips the materialized mask, so only is_causal keeps
    attention causal (padded batches always carry a mask and never exercise it)."""
    single = tokenizer(CAPTIONS[:1], return_tensors="pt")
    truncated = tokenizer(
        [CAPTIONS[0], CAPTIONS[2]], max_length=8, truncation=True, padding="max_length", return_tensors="pt"
    )
    return [(enc["input_ids"], enc["attention_mask"]) for enc in (single, truncated)]


def pixels(batch: int = 3, size: int = 224) -> torch.Tensor:
    return torch.randn(batch, 3, size, size, generator=torch.Generator().manual_seed(1))


def check_equivalence(clip, vision, text, tokenizer):
    px = pixels()
    ids, am = text_inputs(tokenizer)
    with torch.no_grad():
        # encode() without a mask equals HF's own hidden states
        assert torch.allclose(vision.encode(px), clip.vision_model(pixel_values=px).last_hidden_state, atol=1e-5)
        assert torch.allclose(
            text.encode(ids, am), clip.text_model(input_ids=ids, attention_mask=am).last_hidden_state, atol=1e-5
        )
        # embed() equals HF's projected features, normalized
        ref_img = F.normalize(clip.get_image_features(pixel_values=px).pooler_output, dim=-1)
        ref_txt = F.normalize(clip.get_text_features(input_ids=ids, attention_mask=am).pooler_output, dim=-1)
        assert torch.allclose(vision.embed(px), ref_img, atol=1e-5)
        assert torch.allclose(text.embed(ids, am), ref_txt, atol=1e-5)
        # unpadded text matches HF too (guards the causal flag, which padded inputs never reach)
        for u_ids, u_am in unpadded_text_inputs(tokenizer):
            assert u_am.all()
            ref_hidden = clip.text_model(input_ids=u_ids, attention_mask=u_am).last_hidden_state
            assert torch.allclose(text.encode(u_ids, u_am), ref_hidden, atol=1e-5)
            ref_u = F.normalize(clip.get_text_features(input_ids=u_ids, attention_mask=u_am).pooler_output, dim=-1)
            assert torch.allclose(text.embed(u_ids, u_am), ref_u, atol=1e-5)
        # masked paths with nothing masked equal the clean paths
        all_ids = torch.arange(vision.num_patches).expand(px.shape[0], -1)
        assert torch.allclose(vision.encode(px, all_ids), vision.encode(px), atol=1e-5)
        no_mask = torch.zeros_like(ids, dtype=torch.bool)
        assert torch.allclose(text.encode(ids, am, no_mask), text.encode(ids, am), atol=1e-6)


def test_tiny_towers_match_hf(tokenizer):
    check_equivalence(*towers_for(TINY_CLIP), tokenizer)


@pytest.mark.slow
def test_real_clip_towers_match_hf(tokenizer):
    check_equivalence(*towers_for(CLIP_NAME), tokenizer)


def test_vision_encode_keeps_cls_plus_visible_patches():
    _, vision, _ = towers_for(TINY_CLIP)
    ids_keep = torch.tensor([[0, 5, 9], [1, 2, 48]])
    out = vision.encode(pixels(2), ids_keep)
    assert out.shape == (2, 4, vision.hidden_size)


def test_text_mask_embedding_replaces_masked_tokens_and_gets_gradient(tokenizer):
    _, _, text = towers_for(TINY_CLIP)
    ids, am = text_inputs(tokenizer)
    token_mask = torch.zeros_like(ids, dtype=torch.bool)
    token_mask[:, 1] = True  # first word of every caption
    masked = text.encode(ids, am, token_mask)
    assert not torch.allclose(masked, text.encode(ids, am))
    masked.sum().backward()
    assert text.mask_embedding.grad is not None and text.mask_embedding.grad.abs().sum() > 0


def test_mean_pooling_embeds_are_normalized_and_use_new_projection(tokenizer):
    torch.manual_seed(0)
    clip = load_clip(TINY_CLIP).eval()
    vision, text = ClipVisionTower(clip, pooling="mean").eval(), ClipTextTower(clip, pooling="mean").eval()
    assert vision.mean_projection is not None and text.mean_projection is not None
    px = pixels()
    ids, am = text_inputs(tokenizer)
    with torch.no_grad():
        img, txt = vision.embed(px), text.embed(ids, am)
        assert torch.allclose(img.norm(dim=-1), torch.ones(3)) and torch.allclose(txt.norm(dim=-1), torch.ones(3))
        # image: mean over CLS and all patches, through the new projection
        ref_img = F.normalize(vision.mean_projection(vision.encode(px).mean(dim=1)), dim=-1)
        assert torch.allclose(img, ref_img, atol=1e-6)
        # text: mean over non-padded positions only, through the new projection
        tokens = text.encode(ids, am)
        pooled = torch.stack([tokens[i][am[i].bool()].mean(dim=0) for i in range(len(ids))])
        assert torch.allclose(txt, F.normalize(text.mean_projection(pooled), dim=-1), atol=1e-6)
        # extra padding changes nothing: causal attention keeps real-token states and pads are left out
        ids24, am24 = text_inputs(tokenizer, max_len=24)
        assert torch.allclose(text.embed(ids24, am24), txt, atol=1e-5)


def test_parameter_groups():
    _, vision, text = towers_for(TINY_CLIP)
    pretrained_text = {id(p) for p in text.pretrained_parameters()}
    assert id(text.mask_embedding) not in pretrained_text
    assert {id(p) for p in text.native_head_parameters()} == {id(p) for p in text.projection.parameters()}
    head = {id(p) for p in vision.native_head_parameters()}
    assert {id(p) for p in vision.projection.parameters()} <= head
    assert {id(p) for p in vision.model.post_layernorm.parameters()} <= head


def test_build_backbone_registry():
    towers = build_backbone("hf_clip", TINY_CLIP)
    assert isinstance(towers.vision, ClipVisionTower) and isinstance(towers.text, ClipTextTower)
    assert towers.logit_scale.ndim == 0 and towers.logit_scale.requires_grad
    with pytest.raises(ValueError):
        build_backbone("open_clip", TINY_CLIP)
    with pytest.raises(ValueError):
        build_backbone("hf_clip", TINY_CLIP, pooling="max")


@pytest.mark.parametrize("pooling", ["native", "mean"])
def test_pool_of_encode_is_embed(pooling, tokenizer):
    from helpers import make_batch
    from mmae.models.backbones import build_backbone

    torch.manual_seed(0)
    towers = build_backbone("hf_clip", TINY_CLIP, pooling)
    batch = make_batch(tokenizer)
    with torch.no_grad():
        image = towers.vision.pool(towers.vision.encode(batch["pixel_values"]))
        text_tokens = towers.text.encode(batch["input_ids"], batch["attention_mask"])
        text = towers.text.pool(text_tokens, batch["input_ids"], batch["attention_mask"])
        assert torch.equal(image, towers.vision.embed(batch["pixel_values"]))
        assert torch.equal(text, towers.text.embed(batch["input_ids"], batch["attention_mask"]))
