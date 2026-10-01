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
    clip = load_clip(TINY_CLIP)
    vision, text = ClipVisionTower(clip, pooling="mean"), ClipTextTower(clip, pooling="mean")
    ids, am = text_inputs(tokenizer)
    img, txt = vision.embed(pixels()), text.embed(ids, am)
    assert torch.allclose(img.norm(dim=-1), torch.ones(3)) and torch.allclose(txt.norm(dim=-1), torch.ones(3))
    assert vision.mean_projection is not None and text.mean_projection is not None


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
