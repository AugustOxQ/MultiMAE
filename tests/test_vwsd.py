import json

import pytest
import torch

from helpers import CLIP_NAME, make_fake_vwsd, run_train
from mmae.engine.vwsd import VwsdItem, evaluate_vwsd, read_vwsd, vwsd_metrics


def test_read_vwsd(tmp_path):
    items = read_vwsd(make_fake_vwsd(tmp_path / "vwsd"))
    assert len(items) == 3
    assert items[0] == VwsdItem("goal", "football goal", tuple(f"image.{i}." + ("png" if i % 3 == 1 else "jpg") for i in range(10)), "image.3.jpg")


def test_read_vwsd_rejects_a_gold_outside_the_candidates(tmp_path):
    root = make_fake_vwsd(tmp_path / "vwsd")
    (root / "en.test.gold.v1.1.txt").write_text("image.11.jpg\nimage.11.jpg\nimage.1.png\n")
    with pytest.raises(ValueError, match="line 1"):
        read_vwsd(root)


def test_read_vwsd_rejects_mismatched_files(tmp_path):
    root = make_fake_vwsd(tmp_path / "vwsd")
    (root / "en.test.gold.v1.1.txt").write_text("image.3.jpg\n")
    with pytest.raises(ValueError, match="3 items but 1 gold"):
        read_vwsd(root)


def test_vwsd_metrics_ranks():
    items = [VwsdItem("w", "p", ("a", "b", "c"), "a"), VwsdItem("w", "p", ("a", "b", "c"), "b")]
    index = {"a": 0, "b": 1, "c": 2}
    image_emb = torch.eye(3)
    text_emb = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])  # gold "a" ranks 1st, gold "b" 2nd
    metrics = vwsd_metrics(image_emb, text_emb, items, index)
    assert metrics == {"vwsd/hit1": 50.0, "vwsd/mrr": 75.0, "vwsd/n": 2.0}


def test_evaluate_vwsd_on_the_tiny_model(tmp_path):
    from helpers import compose_cfg
    from mmae.data import Collator, build_image_transform
    from mmae.models import MultiMAE

    cfg = compose_cfg("model=contrastive", "model.backbone.pretrained=tiny-random-clip")
    model = MultiMAE(cfg.model, max_text_len=32)
    metrics = evaluate_vwsd(model, make_fake_vwsd(tmp_path / "vwsd"), build_image_transform(CLIP_NAME),
                            Collator(CLIP_NAME, 32), batch_size=4)
    assert metrics["vwsd/n"] == 3.0
    assert 0.0 <= metrics["vwsd/hit1"] <= 100.0 and 10.0 <= metrics["vwsd/mrr"] <= 100.0


def test_evaluate_py_adds_vwsd(tmp_path, fake_coco):
    vwsd = make_fake_vwsd(tmp_path / "vwsd")
    out = tmp_path / "metrics.json"
    result = run_train(tmp_path, fake_coco, "model=contrastive", f"eval.vwsd_dir={vwsd}", f"eval.output={out}",
                       script="evaluate.py")
    assert result.returncode == 0, result.stderr[-3000:]
    metrics = json.loads(out.read_text())
    assert metrics["vwsd/n"] == 3.0 and "rsum" in metrics
