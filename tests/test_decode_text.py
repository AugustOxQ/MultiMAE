"""MultiMAE.decode_text reproduces forward's caption decoder for the same masks (H-b readouts, plan 2 task 1)."""
import pytest
import torch

from helpers import add_emotion, make_batch
from test_model import recorder, tiny_model

ARMS = {
    "ml80": ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0"),
    "parcap": ("model.emotion_head=true", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean",
               "model.loss.weights.mae=0"),
    "coco_ml15": (),
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_decode_text_matches_forward(arm, tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", *ARMS[arm]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    if arm != "coco_ml15":
        batch = add_emotion(batch)
    captured, _, _ = run(batch)
    ids_keep = None if arm == "parcap" else masks["ids_keep"]
    with torch.no_grad():
        logits, emotion = model.decode_text(batch["pixel_values"], batch["input_ids"], batch["attention_mask"],
                                            masks["token"], ids_keep=ids_keep)
    torch.testing.assert_close(logits, captured["text_decoder"], rtol=0, atol=1e-5)
    if arm == "coco_ml15":
        assert emotion is None
    else:
        torch.testing.assert_close(emotion, captured["text_decoder_prefix"][:, 0], rtol=0, atol=1e-5)


def test_decode_text_rejects_models_without_a_text_decoder():
    with pytest.raises(ValueError, match="decode_text"):
        tiny_model("contrastive").decode_text(None, None, None, None)
