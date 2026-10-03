import pytest
import torch
from omegaconf import OmegaConf

from mmae.models.decoders import QueryDecoder
from mmae.models.fusion import ConcatFusion, MultiLearnerFusion, NoFusion, build_fusion

DIM = 16


def fusion_cfg(kind: str, depth: int = 0):
    return OmegaConf.create(dict(type=kind, dim=DIM, depth=depth, heads=4, learner_depth=2, learner_ff_dim=32))


def inputs(batch: int = 3):
    g = torch.Generator().manual_seed(0)
    image = torch.randn(batch, 5, DIM, generator=g)
    text = torch.randn(batch, 8, DIM, generator=g)
    lengths = torch.tensor([8, 5, 3])
    padding = torch.arange(8).unsqueeze(0) >= lengths.unsqueeze(1)
    return image, text, padding


@pytest.mark.parametrize("kind,depth", [("none", 0), ("concat", 0), ("concat", 2), ("multilearner", 0)])
def test_fusion_shapes(kind, depth):
    image, text, padding = inputs()
    out = build_fusion(fusion_cfg(kind, depth)).eval()(image, text, padding)
    if kind == "none":
        assert out.image_memory.shape == (3, 5, DIM) and out.image_padding is None
        assert out.text_memory.shape == (3, 8, DIM) and torch.equal(out.text_padding, padding)
    else:
        assert out.image_memory.shape == (3, 13, DIM) and out.text_memory.shape == (3, 13, DIM)
        assert torch.equal(out.image_padding[:, 5:], padding) and not out.image_padding[:, :5].any()


@pytest.mark.parametrize("kind", ["concat", "multilearner"])
def test_fusion_handles_one_modality(kind):
    image, text, padding = inputs()
    fusion = build_fusion(fusion_cfg(kind)).eval()
    assert fusion(image, None, None).image_memory.shape == (3, 5, DIM)
    assert fusion(None, text, padding).text_memory.shape == (3, 8, DIM)


def test_multilearner_gives_each_decoder_its_own_memory():
    image, text, padding = inputs()
    out = MultiLearnerFusion(DIM, heads=4, learner_depth=1, learner_ff_dim=32).eval()(image, text, padding)
    assert not torch.allclose(out.image_memory, out.text_memory)


def test_unknown_fusion_type():
    with pytest.raises(ValueError):
        build_fusion(fusion_cfg("cross_attention"))


@pytest.mark.parametrize("kind,depth", [("none", 0), ("concat", 0), ("concat", 2), ("multilearner", 0)])
def test_padded_text_positions_do_not_change_any_output(kind, depth):
    torch.manual_seed(0)
    fusion = build_fusion(fusion_cfg(kind, depth)).eval()
    image_decoder = QueryDecoder(7, DIM, 12, depth=2, heads=4).eval()
    text_decoder = QueryDecoder(8, DIM, 50, depth=2, heads=4).eval()
    image, text, padding = inputs()
    noisy = text.clone()
    noisy[padding] = torch.randn(int(padding.sum()), DIM) * 10  # garbage at padded positions only

    def run(text_tokens):
        out = fusion(image, text_tokens, padding)
        img = image_decoder(out.image_memory, out.image_padding)
        txt = text_decoder(out.text_memory, out.text_padding, query_padding=padding)
        return img, txt

    with torch.no_grad():
        img_a, txt_a = run(text)
        img_b, txt_b = run(noisy)
    assert torch.allclose(img_a, img_b, atol=1e-5)
    assert torch.allclose(txt_a[~padding], txt_b[~padding], atol=1e-5)


def test_query_decoder_shapes_and_query_count():
    decoder = QueryDecoder(10, DIM, 6, depth=1, heads=4)
    memory = torch.randn(2, 4, DIM)
    assert decoder(memory).shape == (2, 10, 6)
    assert decoder(memory, query_padding=torch.zeros(2, 7, dtype=torch.bool)).shape == (2, 7, 6)
    with pytest.raises(ValueError):
        decoder(memory, query_padding=torch.zeros(2, 11, dtype=torch.bool))


def test_append_memory_token_adds_one_visible_position():
    from mmae.models.fusion import append_memory_token

    memory = torch.randn(2, 5, 8)
    token = torch.randn(2, 8)
    out, padding = append_memory_token(memory, None, token)
    assert out.shape == (2, 6, 8) and torch.equal(out[:, -1], token) and torch.equal(out[:, :5], memory)
    assert padding.shape == (2, 6) and not padding.any()
    given = torch.tensor([[False] * 4 + [True], [False] * 5])
    _, padding = append_memory_token(memory, given, token)
    assert torch.equal(padding[:, :5], given) and not padding[:, 5].any()
