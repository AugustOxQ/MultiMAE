"""ArtELingo wiring: collate, dataset factory, configs, and a CPU smoke of every D4 arm (spec 2026-10-07)."""
import json

import pytest
import torch

from helpers import compose_cfg, run_train_artelingo
from mmae.data import Collator
from mmae.data.factory import build_pairs, build_retrieval, dataset_name
from mmae.data.transforms import build_image_transform

ARMS = {
    "C": ("model=contrastive",),
    "ML-80": ("model=fusion_multilearner", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0", "model.emotion_head=true"),
    "ML-80+MAE": ("model=fusion_multilearner", "model.masking.text_ratio=0.8", "model.emotion_head=true"),
    "Par-cap": ("model=fusion_multilearner", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean",
                "model.loss.weights.mae=0", "model.emotion_head=true"),
}


def test_configs():
    local, cluster = compose_cfg("data=artelingo"), compose_cfg("data=artelingo_cluster")
    assert set(local.data) == set(cluster.data)
    assert local.data.name == "artelingo" and local.data.max_text_len == 40 == cluster.data.max_text_len
    assert local.eval.extended_metrics is False and cluster.eval.extended_metrics is False
    assert cluster.data.images_dir == "/local/wding/Dataset/wikiart_proj/wikiart"
    assert cluster.data.annotations_dir == "/local/wding/Dataset/artelingo_mmae"
    assert cluster.paths.res_dir == "/local/wding/res/MultiMAE/artelingo"
    assert dataset_name(compose_cfg().data) == "coco"  # COCO configs carry no name


def test_collate_adds_emotion_for_artelingo_pairs():
    collator = Collator("openai/clip-vit-base-patch32", 40)
    image = torch.zeros(3, 224, 224)
    batch = collator.pairs([(image, "a calm sea", 2), (image, "an angry sky", 4)])
    assert batch["emotion"].tolist() == [2, 4] and batch["emotion"].dtype == torch.long
    assert "emotion" not in collator.pairs([(image, "a dog"), (image, "a cat")])


def test_factory_builds_both_datasets(fake_artelingo, fake_coco):
    images, annotations, heldout = fake_artelingo
    transform = build_image_transform("openai/clip-vit-base-patch32")
    dcfg = compose_cfg("data=artelingo", f"data.images_dir={images}", f"data.annotations_dir={annotations}",
                       f"data.heldout_file={heldout}").data
    assert len(build_pairs(dcfg, "train", transform, None)) == 10
    assert len(build_retrieval(dcfg, "val", transform, None)) == 2
    coco_images, coco_annotations = fake_coco
    ccfg = compose_cfg(f"data.images_dir={coco_images}", f"data.annotations_dir={coco_annotations}").data
    assert len(build_pairs(ccfg, "train", transform, None)) == 16


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_trains_on_artelingo(arm, tmp_path, fake_artelingo):
    result = run_train_artelingo(tmp_path, fake_artelingo, *ARMS[arm], "train.seeded_sampler=true")
    assert result.returncode == 0, result.stderr[-3000:]
    (run_json,) = list((tmp_path / "res").rglob("run.json"))
    run = json.loads(run_json.read_text())
    assert run["status"] == "completed"
    test = run["results"]["test"]
    assert "test/retrieval/rsum" in test
    assert not any(k.startswith("test/eccv") for k in test)
    if arm != "C":
        assert "test/loss_emotion" in test and "test/loss_mlm_tokens" in test


def test_extended_metrics_guard_skips_artelingo(fake_artelingo):
    from mmae.engine.eccv import build_extended_metrics

    images, annotations, heldout = fake_artelingo
    cfg = compose_cfg("data=artelingo", f"data.annotations_dir={annotations}", "eval.extended_metrics=true")
    assert build_extended_metrics(cfg, "test", True) is None
