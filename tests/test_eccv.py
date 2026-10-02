"""Extended COCO test metrics (mmae.engine.eccv): id mapping, truncated rankings against full rankings on a
synthetic problem, the gate, and checks against the real annotation files (skipped when they are absent)."""
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from eccv_caption import Metrics
from eccv_caption._metrics import rprecision

from helpers import compose_cfg
from mmae.data.coco import retrieval_items
from mmae.engine import eccv
from mmae.engine.eccv import (
    PM_FILES,
    build_extended_metrics,
    check_ids,
    coco_1k_folds,
    coco_image_id,
    coco_test_metrics,
    map_coco_ids,
)
from mmae.engine.retrieval import retrieval_metrics

TABLE4_KEYS = {"eccv/map_at_r", "eccv/rprecision", "eccv/r1", "cxc/r1", "coco1k/r1", "coco5k/r1", "pmrp"}
ALL_TARGETS = ("coco_1k_recalls", "coco_5k_recalls", "cxc_recalls", "eccv_map_at_r", "eccv_rprecision", "eccv_r1")


# ---------------------------------------------------------------- id mapping

def write_captions(path: Path, annotations) -> Path:
    path.write_text(json.dumps({"annotations": [{"image_id": i, "id": a, "caption": t} for i, a, t in annotations]}))
    return path


def test_coco_image_id():
    assert coco_image_id("val2014/COCO_val2014_000000391895.jpg") == 391895
    assert coco_image_id("COCO_val2014_000000000042.jpg") == 42
    with pytest.raises(ValueError, match="no COCO image id"):
        coco_image_id("val2014/cat.jpg")


def test_map_coco_ids_matches_text_per_image_and_repeated_texts_in_order(tmp_path):
    captions = write_captions(tmp_path / "captions_val2014.json", [
        (7, 70, "a dog"), (7, 71, "a cat"), (9, 90, "a dog"), (7, 72, "a dog"), (9, 91, "a bird"), (9, 92, "a fish"),
    ])
    items = [
        ("val2014/COCO_val2014_000000000009.jpg", ["a bird", "a dog", "a fish"]),
        ("val2014/COCO_val2014_000000000007.jpg", ["a dog", "a cat", "a dog"]),  # one text twice: two annotations
    ]
    image_ids, caption_ids = map_coco_ids(items, captions)
    assert image_ids.tolist() == [9, 7]
    assert caption_ids.tolist() == [[91, 90, 92], [70, 71, 72]]  # "a dog" of image 9 is 90, not 70 or 72


@pytest.mark.parametrize("bad", [
    ["a dog", "a cat", "a dog", "a dog"],  # a third copy: both "a dog" annotations are taken
    ["a dog", "a cat", "a dog "],  # text must match exactly
    ["a dog", "a cat", "a bird"],  # a caption of another image
])
def test_map_coco_ids_rejects_unmatched_captions(tmp_path, bad):
    captions = write_captions(tmp_path / "captions_val2014.json", [
        (7, 70, "a dog"), (7, 71, "a cat"), (7, 72, "a dog"), (9, 91, "a bird"),
    ])
    with pytest.raises(ValueError, match="matches no unused annotation of COCO image 7"):
        map_coco_ids([("val2014/COCO_val2014_000000000007.jpg", bad)], captions)


# ---------------------------------------------------------------- synthetic problem

def synthetic_problem(seed: int, embeddings: str, n_images: int = 120, k: int = 5, dim: int = 6):
    """Embeddings in dataset order, their COCO-like ids, and a Metrics whose ground truths are replaced by
    synthetic ones shaped like the real ones: COCO folds in a shuffled image order, CxC and ECCV positives
    beyond the original pairs (ECCV lists longer than 10, one id outside the test set), PM positives
    without the original pairs, some lists longer than the PMRP cut and some empty."""
    rng = np.random.default_rng(seed)
    image_ids = rng.choice(np.arange(100_000, 200_000), n_images, replace=False)
    caption_ids = rng.choice(np.arange(500_000, 900_000), n_images * k, replace=False).reshape(n_images, k)
    # Integer embeddings make every similarity exact in float32, so ties are real ties (and reproducible).
    if embeddings == "ties":  # few distinct values: many tied similarities
        images = rng.integers(-1, 2, (n_images, dim))
        captions = images[:, None] + rng.integers(-1, 2, (n_images, k, dim))
    elif embeddings == "spread":  # almost no ties
        images = rng.integers(-60, 61, (n_images, dim))
        captions = images[:, None] + rng.integers(-60, 61, (n_images, k, dim))
    else:  # continuous, for comparisons with retrieval_metrics (which counts tied candidates as ranked above)
        images = rng.normal(size=(n_images, dim))
        captions = images[:, None] + 1.5 * rng.normal(size=(n_images, k, dim))
    images, captions = torch.tensor(images, dtype=torch.float32), torch.tensor(captions, dtype=torch.float32)
    if embeddings == "ties":
        captions[3, 1] = captions[10, 2]  # one text under two images
        images[5] = images[6]  # two identical images

    flat = caption_ids.reshape(-1)
    owner = {int(c): int(i) for i, row in zip(image_ids, caption_ids) for c in row}
    own = {int(i): [int(c) for c in row] for i, row in zip(image_ids, caption_ids)}

    def sample(pool, size, exclude=()):
        pool = [int(x) for x in pool if int(x) not in set(exclude)]
        return [int(x) for x in rng.choice(pool, min(size, len(pool)), replace=False)]

    metrics = Metrics()
    metrics.coco_ids = np.array(
        [c for i in rng.permutation(n_images) for c in rng.permutation(caption_ids[i])], dtype=np.int64
    )
    metrics.coco_gts = {"i2t": own, "t2i": {c: [i] for c, i in owner.items()}}
    metrics.cxc_gts = {
        "i2t": {i: own[i][: rng.integers(3, k + 1)] + sample(flat, rng.integers(0, 4), own[i]) for i in own},
        "t2i": {c: [i] + sample(image_ids, rng.integers(0, 3), [i]) for c, i in owner.items() if rng.random() < 0.9},
    }
    metrics.eccv_gts = {
        "i2t": {i: own[i] + sample(flat, rng.integers(0, 16), own[i]) for i in own if rng.random() < 0.5},
        "t2i": {c: [i] + sample(image_ids, rng.integers(0, 13), [i]) for c, i in owner.items() if rng.random() < 0.3},
    }
    next(iter(metrics.eccv_gts["i2t"].values())).append(999_999)  # a positive outside the test set, as in the real file
    metrics.pm_gts = {
        "i2t": {i: sample(flat, rng.integers(1, 300), own[i]) for i in own if rng.random() < 0.9},
        "t2i": {c: sample(image_ids, rng.integers(0, 80), [i]) for c, i in owner.items() if rng.random() < 0.95},
    }
    assert any(not p for p in metrics.pm_gts["t2i"].values())  # empty PM lists, as in the real files
    return images, captions, image_ids.astype(np.int64), caption_ids.astype(np.int64), metrics


def full_rankings(images, captions, image_ids, caption_ids):
    """Every candidate per query, by similarity, ties in dataset order."""
    n, k, d = captions.shape
    scores = images @ captions.reshape(n * k, d).T
    i2t = torch.sort(scores, dim=1, descending=True, stable=True).indices
    t2i = torch.sort(scores.T, dim=1, descending=True, stable=True).indices
    flat = caption_ids.reshape(-1).tolist()
    return {
        "i2t": {int(q): [flat[j] for j in row] for q, row in zip(image_ids, i2t.tolist())},
        "t2i": {int(q): [int(image_ids[j]) for j in row] for q, row in zip(flat, t2i.tolist())},
    }


def reference_metrics(images, captions, image_ids, caption_ids, metrics, pmrp_max_r):
    """The package's metrics on full rankings; PMRP as the ECCV Caption paper defines it, R = min(R, cut)."""
    full = full_rankings(images, captions, image_ids, caption_ids)
    scores = metrics.compute_all_metrics(full["i2t"], full["t2i"], target_metrics=ALL_TARGETS, Ks=(1, 5, 10))
    scores["pmrp"] = {
        d: np.mean([rprecision(full[d][q], set(p), min(len(set(p)), pmrp_max_r)) for q, p in metrics.pm_gts[d].items() if p])
        for d in ("i2t", "t2i")
    }
    return eccv._flatten(scores)


@pytest.mark.parametrize("embeddings", ["ties", "spread"])
@pytest.mark.parametrize("pmrp_max_r,chunk", [(50, 7), (50, 1000), (7, 64)])
def test_truncated_rankings_give_the_full_ranking_metrics(embeddings, pmrp_max_r, chunk):
    images, captions, image_ids, caption_ids, metrics = synthetic_problem(0, embeddings)
    eccv_max = max(len(p) for d in ("i2t", "t2i") for p in metrics.eccv_gts[d].values())
    assert 10 < eccv_max < 50 and max(map(len, metrics.pm_gts["i2t"].values())) > 50  # truncation matters
    expected = reference_metrics(images, captions, image_ids, caption_ids, metrics, pmrp_max_r)
    got = coco_test_metrics(images, captions, image_ids, caption_ids, metrics, pmrp_max_r=pmrp_max_r, chunk=chunk)
    assert got == expected
    assert 5 < got["coco1k/r1"] < 95 and 5 < got["eccv/map_at_r"] < 95 and 5 < got["pmrp"] < 95  # not degenerate


def test_metric_keys():
    images, captions, image_ids, caption_ids, metrics = synthetic_problem(1, "spread")
    got = coco_test_metrics(images, captions, image_ids, caption_ids, metrics)
    per_direction = {f"eccv/{d}_{m}" for d in ("i2t", "t2i") for m in ("map_at_r", "rprecision", "r1")}
    per_direction |= {f"{p}/{d}_r{K}" for p in ("cxc", "coco1k", "coco5k") for d in ("i2t", "t2i") for K in (1, 5, 10)}
    per_direction |= {"pmrp/i2t", "pmrp/t2i"}
    assert set(got) == per_direction | TABLE4_KEYS | {"coco1k/rsum"}
    assert not any("@" in key for key in got)
    for key in TABLE4_KEYS - {"pmrp"}:
        prefix, name = key.split("/")
        assert got[key] == pytest.approx((got[f"{prefix}/i2t_{name}"] + got[f"{prefix}/t2i_{name}"]) / 2)
    assert got["pmrp"] == pytest.approx((got["pmrp/i2t"] + got["pmrp/t2i"]) / 2)
    metrics.pm_gts = {}  # no PM ground truth (data.pm_dir null): no PMRP keys
    assert not any(key.startswith("pmrp") for key in coco_test_metrics(images, captions, image_ids, caption_ids, metrics))


def test_coco_recalls_agree_with_retrieval_metrics():
    """Package COCO 5K recalls equal our own retrieval_metrics; COCO 1K recalls equal retrieval_metrics run
    on each fold's images and captions, averaged (continuous embeddings: no ties)."""
    images, captions, image_ids, caption_ids, metrics = synthetic_problem(2, "continuous")
    got = coco_test_metrics(images, captions, image_ids, caption_ids, metrics)
    ours = retrieval_metrics(images, captions)
    image_fold, _ = coco_1k_folds(metrics, image_ids, caption_ids)
    folds = [retrieval_metrics(images[image_fold == f], captions[image_fold == f]) for f in range(5)]
    for d in ("i2t", "t2i"):
        for K in (1, 5, 10):
            assert got[f"coco5k/{d}_r{K}"] == pytest.approx(ours[f"{d}_R{K}"], abs=1e-9)
            assert got[f"coco1k/{d}_r{K}"] == pytest.approx(np.mean([m[f"{d}_R{K}"] for m in folds]), abs=1e-9)


def test_package_pmrp_divides_by_zero_on_empty_positives():
    """Why queries with no plausible match are left out: the package's R-Precision divides by zero on them
    (eccv-caption issue #2). If an upgrade changes this, revisit coco_test_metrics."""
    metrics = Metrics()
    metrics.pm_gts = {"i2t": {1: [10]}, "t2i": {10: [1], 11: []}}
    with pytest.raises(ZeroDivisionError):
        metrics.pmrp({"i2t": {1: [10, 11]}, "t2i": {10: [1], 11: [1]}}, "all")


def test_check_ids():
    images, captions, image_ids, caption_ids, metrics = synthetic_problem(3, "spread")
    check_ids(image_ids, caption_ids, metrics)
    swapped = caption_ids.copy()
    swapped[0, 0], swapped[1, 0] = caption_ids[1, 0], caption_ids[0, 0]  # same set, wrong images
    with pytest.raises(ValueError, match="belong to another image"):
        check_ids(image_ids, swapped, metrics)
    missing = caption_ids.copy()
    missing[0, 0] = 1
    with pytest.raises(ValueError, match="1 extra, 1 missing"):
        check_ids(image_ids, missing, metrics)
    with pytest.raises(ValueError, match="1 extra, 1 missing"):
        coco_test_metrics(images, captions, image_ids, missing, metrics)  # an unknown id has no fold


def test_images_split_across_folds_are_rejected():
    images, captions, image_ids, caption_ids, metrics = synthetic_problem(4, "spread")
    ids = metrics.coco_ids.copy()
    size = len(ids) // 5
    ids[size - 1], ids[size] = ids[size], ids[size - 1]  # two images now straddle folds 0 and 1
    metrics.coco_ids = ids
    with pytest.raises(ValueError, match="2 images have captions in two COCO 1K folds"):
        coco_test_metrics(images, captions, image_ids, caption_ids, metrics)


# ---------------------------------------------------------------- the gate

def test_gate_skips_unless_full_test_split_of_two_modalities(fake_coco, caplog):
    images_dir, annotations_dir = fake_coco
    base = [f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}"]
    on = compose_cfg(*base, "eval.extended_metrics=true")
    caplog.set_level(logging.INFO, logger="mmae.engine.eccv")
    assert build_extended_metrics(compose_cfg(*base, "eval.extended_metrics=false"), "test", True) is None
    assert build_extended_metrics(on, "val", True) is None
    assert build_extended_metrics(on, "test", False) is None
    assert not [r for r in caplog.records if r.name == "mmae.engine.eccv"]  # the silent cases
    assert build_extended_metrics(compose_cfg(*base, "eval.extended_metrics=true", "data.limit_test=3"), "test", True) is None
    assert "data.limit_test=3" in caplog.text
    assert build_extended_metrics(on, "test", True) is None  # the fake COCO's test split has 6 images
    assert "the test split has 6 images, not COCO's 5000" in caplog.text


# ---------------------------------------------------------------- real annotation files (CPU, JSON only)

REAL = compose_cfg().data
needs_real_coco = pytest.mark.skipif(
    not (Path(REAL.annotations_dir) / eccv.CAPTIONS_FILE).is_file()
    or not (Path(REAL.annotations_dir) / "coco_karpathy_test.json").is_file(),
    reason="needs the real COCO annotations (data/coco.yaml)",
)
needs_real_pm = pytest.mark.skipif(
    not all((Path(REAL.pm_dir) / name).is_file() for name in PM_FILES), reason="needs the PM files (data.pm_dir)"
)


@needs_real_coco
def test_real_karpathy_test_split_maps_to_the_package_ids():
    extended = build_extended_metrics(compose_cfg("data.pm_dir=null"), "test", two_modalities=True)
    assert extended is not None
    package = Metrics()
    assert sorted(extended.caption_ids.reshape(-1).tolist()) == sorted(package.coco_ids.tolist())
    for image_id, row in zip(extended.image_ids.tolist(), extended.caption_ids.tolist()):
        assert set(row) == set(package.coco_gts["i2t"][image_id])
    items = dict(retrieval_items(REAL.annotations_dir, "test"))
    for image_id in (441453, 576369, 26654):  # one text listed twice, two COCO annotations
        image = next(name for name in items if coco_image_id(name) == image_id)
        assert len(set(items[image])) == 4
        assert len(set(extended.caption_ids[extended.image_ids == image_id][0].tolist())) == 5


@needs_real_coco
def test_real_missing_pm_files_fail_at_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="pm_image_to_caption.json"):
        build_extended_metrics(compose_cfg(f"data.pm_dir={tmp_path}"), "test", two_modalities=True)


@needs_real_coco
@needs_real_pm
def test_real_ground_truth_with_random_embeddings(caplog):
    """End to end on the real ground truth (PM included) with random embeddings: every key, COCO 5K and 1K
    against retrieval_metrics, and the 1,130 empty PM lists left out."""
    extended = build_extended_metrics(compose_cfg(), "test", two_modalities=True)
    g = torch.Generator().manual_seed(0)
    images = torch.nn.functional.normalize(torch.randn(5000, 64, generator=g), dim=-1)
    captions = torch.nn.functional.normalize(images[:, None] + 0.25 * torch.randn(5000, 5, 64, generator=g), dim=-1)
    caplog.set_level(logging.INFO, logger="mmae.engine.eccv")
    got = extended(images, captions)
    assert "left out 0 i2t and 1130 t2i queries" in caplog.text
    assert TABLE4_KEYS | {"pmrp/i2t", "pmrp/t2i", "coco1k/rsum"} <= set(got) and len(got) == 34
    assert all(0.0 <= v <= 100.0 for k, v in got.items() if k != "coco1k/rsum")
    ours = retrieval_metrics(images, captions)
    image_fold, _ = coco_1k_folds(Metrics(), extended.image_ids, extended.caption_ids)
    folds = [retrieval_metrics(images[image_fold == f], captions[image_fold == f]) for f in range(5)]
    for d in ("i2t", "t2i"):
        for K in (1, 5, 10):
            assert got[f"coco5k/{d}_r{K}"] == pytest.approx(ours[f"{d}_R{K}"], abs=1e-9)
            assert got[f"coco1k/{d}_r{K}"] == pytest.approx(np.mean([m[f"{d}_R{K}"] for m in folds]), abs=1e-9)
    assert 20 < got["coco5k/r1"] < 99  # the embeddings carry signal: the check is not on chance-level ranks
