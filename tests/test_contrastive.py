import pytest
import torch
from omegaconf import open_dict

from helpers import make_batch
from mmae.losses import contrastive_loss
from test_model import tiny_model

HEAD_PREFIXES = ("image_proj", "text_proj", "image_decoder", "text_decoder", "fusion")


def test_contrastive_has_no_reconstruction_modules_and_returns_two_losses(tokenizer):
    model = tiny_model("contrastive")
    names = [n for n, _ in model.named_parameters()]
    assert not [n for n in names if n.startswith(HEAD_PREFIXES)], names
    assert not any(hasattr(model, attr) for attr in HEAD_PREFIXES)
    out = model(make_batch(tokenizer))
    assert set(out) == {"loss", "loss_contrastive"}
    assert torch.equal(out["loss"], out["loss_contrastive"])


def test_contrastive_loss_matches_embeddings(tokenizer):
    model = tiny_model("contrastive").eval()
    batch = make_batch(tokenizer)
    with torch.no_grad():
        out = model(batch)
        expected = contrastive_loss(
            model.embed_image(batch["pixel_values"]),
            model.embed_text(batch["input_ids"], batch["attention_mask"]),
            model.logit_scale,
            model.gather,
        )
    assert torch.equal(out["loss"], expected)


def test_config_without_reconstruction_key_builds_full_model(tokenizer):
    from helpers import compose_cfg
    from mmae.models import MultiMAE
    from mmae.models.backbones import TINY_CLIP

    cfg = compose_cfg("model=fusion_concat", f"model.backbone.pretrained={TINY_CLIP}")
    with open_dict(cfg.model):
        del cfg.model["reconstruction"]  # a config.yaml saved before the flag existed
    model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
    assert set(model(make_batch(tokenizer))) == {"loss", "loss_contrastive", "loss_mae", "loss_mlm"}


@pytest.mark.parametrize("modalities", ["[image]", "[text]"])
def test_contrastive_needs_both_modalities(modalities):
    with pytest.raises(ValueError, match="both modalities"):
        tiny_model("contrastive", f"model.modalities={modalities}")
