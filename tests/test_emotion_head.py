"""H-b emotion head (spec 2026-10-07, section 5.1): one extra text-decoder query, 9 classes, always a target."""
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from helpers import add_emotion, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP
from mmae.models.decoders import QueryDecoder
from mmae.models.model import NUM_EMOTIONS
from test_model import recorder, tiny_model

ML80 = ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0")


def test_query_decoder_prefix_shapes_and_backward_compatibility():
    torch.manual_seed(0)
    memory = torch.randn(2, 5, 16)
    plain = QueryDecoder(7, 16, 11, depth=1, heads=2)
    assert plain(memory).shape == (2, 7, 11)
    with_prefix = QueryDecoder(7, 16, 11, depth=1, heads=2, prefix_queries=1, prefix_out_dim=9)
    padding = torch.zeros(2, 7, dtype=torch.bool)
    padding[1, 4:] = True
    main, prefix = with_prefix(memory, query_padding=padding)
    assert main.shape == (2, 7, 11) and prefix.shape == (2, 1, 9)
    assert torch.isfinite(main).all() and torch.isfinite(prefix).all()
    assert {"prefix_queries", "prefix_head.weight", "prefix_head.bias"} <= {n for n, _ in with_prefix.named_parameters()}
    assert not any(n.startswith("prefix") for n, _ in plain.named_parameters())


def test_emotion_loss_is_a_separate_weighted_term(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", *ML80).eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_emotion(make_batch(tokenizer))
    captured, out, _ = run(batch)
    token_mask = masks["token"]
    logits, emotion_logits = captured["text_decoder"], captured["text_decoder_prefix"][:, 0]
    token_ce = F.cross_entropy(logits[token_mask].float(), batch["input_ids"][token_mask])
    emotion_ce = F.cross_entropy(emotion_logits.float(), batch["emotion"])
    torch.testing.assert_close(out["loss_mlm"], token_ce)
    torch.testing.assert_close(out["loss_emotion"], emotion_ce)
    weights = model.loss_weights
    assert weights["emotion"] == 0.07
    expected = (weights["contrastive"] * out["loss_contrastive"] + weights["mae"] * out["loss_mae"]
                + weights["mlm"] * out["loss_mlm"] + 0.07 * out["loss_emotion"])
    torch.testing.assert_close(out["loss"], expected)
    assert "loss_mlm_tokens" not in out


def test_emotion_head_needs_an_emotion_weight():
    cfg = compose_cfg("model=fusion_multilearner", f"model.backbone.pretrained={TINY_CLIP}", *ML80)
    plain = OmegaConf.to_container(cfg.model)
    plain["loss"]["weights"].pop("emotion")
    with pytest.raises(ValueError, match="loss.weights.emotion"):
        MultiMAE(OmegaConf.create(plain), max_text_len=cfg.data.max_text_len)


def test_emotion_is_a_target_never_an_input(tokenizer, monkeypatch):
    """Changing the labels moves the loss but no decoder output; the slot reads visible text, not hidden text."""
    model = tiny_model("fusion_multilearner", *ML80).eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_emotion(make_batch(tokenizer))
    base_out, base_loss, _ = run(batch)
    relabelled = dict(batch, emotion=(batch["emotion"] + 1) % NUM_EMOTIONS)
    out, loss, _ = run(relabelled)
    assert torch.equal(out["text_decoder_prefix"], base_out["text_decoder_prefix"])
    assert not torch.equal(loss["loss_emotion"], base_loss["loss_emotion"])
    hidden = masks["token"]
    visible = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool() & ~hidden
    noise = torch.randint(1000, 40000, batch["input_ids"].shape)
    assert torch.equal(run(dict(batch, input_ids=torch.where(hidden, noise, batch["input_ids"])))[0]["text_decoder_prefix"],
                       base_out["text_decoder_prefix"])
    assert not torch.equal(run(dict(batch, input_ids=torch.where(visible, noise, batch["input_ids"])))[0]["text_decoder_prefix"],
                           base_out["text_decoder_prefix"])


def test_emotion_head_needs_labels(tokenizer):
    model = tiny_model("fusion_multilearner", *ML80)
    with pytest.raises(KeyError, match="emotion"):
        model(make_batch(tokenizer))


@pytest.mark.parametrize("name", ["contrastive", "image_mae"])
def test_emotion_head_needs_the_text_decoder(name):
    with pytest.raises(ValueError, match="emotion_head"):
        tiny_model(name, "model.emotion_head=true")


def test_old_configs_build_without_the_key():
    cfg = compose_cfg("model=fusion_multilearner", f"model.backbone.pretrained={TINY_CLIP}")
    old = OmegaConf.to_container(cfg.model)
    old.pop("emotion_head", None)
    model = MultiMAE(OmegaConf.create(old), max_text_len=cfg.data.max_text_len)
    assert model.emotion_head is False
    assert not any("prefix" in n for n, _ in model.named_parameters())


def test_prefix_query_gets_no_weight_decay():
    model = tiny_model("fusion_multilearner", *ML80)
    prefix = model.text_decoder.prefix_queries
    groups = model.param_groups(1e-4, 1e-5, 0.05)
    owner = [g for g in groups if any(p is prefix for p in g["params"])]
    assert len(owner) == 1 and owner[0]["weight_decay"] == 0.0
