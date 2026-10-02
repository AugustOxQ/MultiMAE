import json

import pytest

from helpers import REPO, run_train


def test_evaluate_a_trained_run(tmp_path, fake_coco):
    trained = run_train(tmp_path, fake_coco, "model=fusion_concat", "train.save=best")
    assert trained.returncode == 0, trained.stderr[-5000:]
    run_dir = next((tmp_path / "res").glob("multimae/default/*"))
    result = run_train(tmp_path, fake_coco, f"eval.run_dir={run_dir}", "eval.split=val", script="evaluate.py")
    assert result.returncode == 0, result.stderr[-5000:]
    info = json.loads((run_dir / "run.json").read_text())
    assert set(info["eval"]["val"]) >= {"i2t_R1", "t2i_R1", "rsum"}
    assert info["status"] == "completed"  # evaluation does not touch the training status


def test_evaluate_zero_shot_tiny(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "eval.split=test", script="evaluate.py")
    assert result.returncode == 0, result.stderr[-5000:]
    assert "rsum" in result.stderr + result.stdout


@pytest.mark.parametrize("limit_test", ["null", "3"])
def test_evaluate_skips_extended_metrics_on_fake_coco(tmp_path, fake_coco, limit_test):
    """Switched on, the extended metrics still skip the fake COCO's 6-image test split and any limit_test;
    eval.output's missing parent folders are created."""
    out = tmp_path / "new" / "folder" / "metrics.json"
    result = run_train(
        tmp_path, fake_coco, "eval.split=test", "eval.extended_metrics=true", f"data.limit_test={limit_test}",
        f"eval.output={out}", script="evaluate.py",
    )
    assert result.returncode == 0, result.stderr[-5000:]
    assert "extended test metrics (ECCV Caption, CxC, COCO 1K, PMRP) skipped" in result.stderr
    metrics = json.loads(out.read_text())
    assert "rsum" in metrics and not any("/" in key for key in metrics)


def test_evaluate_rejects_single_modality(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "model=image_mae", script="evaluate.py")
    assert result.returncode != 0 and "both modalities" in result.stderr


@pytest.fixture(scope="module")
def zero_shot_b32(tmp_path_factory):
    """evaluate.py on the real COCO 5k test split with zero-shot CLIP ViT-B/32 (GPU), run once for the module."""
    import os
    import subprocess
    import sys

    cwd = tmp_path_factory.mktemp("zero_shot")
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_USE_CPU")}
    out = cwd / "zero_shot.json"
    cmd = [sys.executable, str(REPO / "evaluate.py"), "eval.split=test", "train.eval_batch_size=256", f"eval.output={out}"]
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-5000:]
    return json.loads(out.read_text())


@pytest.mark.slow
def test_zero_shot_clip_b32_matches_published(zero_shot_b32):
    metrics = zero_shot_b32
    # OpenAI CLIP ViT-B/32 zero-shot, COCO 5k test: i2t R@1 50.1, t2i R@1 30.4 (published)
    assert abs(metrics["i2t_R1"] - 50.1) < 1.5, metrics
    assert abs(metrics["t2i_R1"] - 30.4) < 1.5, metrics


# ECCV Caption paper (Chun et al., ECCV 2022), CLIP ViT-B/32 zero-shot. Table 4 gives the means of i2t and
# t2i, checked within 0.1. Its appendix Tables D.2 (i2t) and D.3 (t2i) give per-direction values, checked
# within 0.2 (PMRP t2i within 0.1) where the paper computes them as we do. Left out: ECCV R@1 per direction
# (one flipped query moves it by about 0.08, and ours differ by -0.16 / +0.23 while the mean agrees);
# COCO 1K R@5 and R@10, which the paper reads from 5K top-50 lists filtered to each fold, under-counting
# them (95.00 / 87.70 at R@10, ours from full rankings of the fold 95.68 / 88.74); and PMRP i2t, whose
# 54.40 in D.2 contradicts Table 4 (2 x 55.32 - 50.69 = 59.95, ours 59.95).
ECCV_TABLE4_CLIP_B32 = {
    "eccv/map_at_r": 26.75, "eccv/rprecision": 36.91, "eccv/r1": 67.08, "cxc/r1": 41.97,
    "coco1k/r1": 59.47, "coco5k/r1": 40.28, "pmrp": 55.32,
}
ECCV_TABLE_D_CLIP_B32 = {
    "eccv/i2t_map_at_r": 22.39, "eccv/i2t_rprecision": 32.61, "cxc/i2t_r1": 51.68, "coco1k/i2t_r1": 69.26,
    "coco5k/i2t_r1": 50.14, "coco5k/i2t_r5": 75.00, "coco5k/i2t_r10": 83.42,
    "eccv/t2i_map_at_r": 31.11, "eccv/t2i_rprecision": 41.20, "cxc/t2i_r1": 32.26, "coco1k/t2i_r1": 49.68,
    "coco5k/t2i_r1": 30.42, "coco5k/t2i_r5": 55.96, "coco5k/t2i_r10": 66.89, "pmrp/t2i": 50.69,
}
ECCV_TOLERANCE = {**{k: 0.1 for k in ECCV_TABLE4_CLIP_B32}, **{k: 0.2 for k in ECCV_TABLE_D_CLIP_B32}, "pmrp/t2i": 0.1}


def eccv_caption_paper_deviations(metrics: dict[str, float]) -> dict[str, tuple[float, float]]:
    """(ours, paper) for every compared value beyond its tolerance; ours is nan for a missing key."""
    expected = {**ECCV_TABLE4_CLIP_B32, **ECCV_TABLE_D_CLIP_B32}
    ours = {k: metrics.get(k, float("nan")) for k in expected}
    return {k: (round(ours[k], 3), v) for k, v in expected.items() if not abs(ours[k] - v) <= ECCV_TOLERANCE[k]}


@pytest.mark.slow
def test_zero_shot_clip_b32_matches_eccv_caption_paper(zero_shot_b32):
    """Table 4 means within 0.1 points, comparable per-direction values within 0.2 (PMRP t2i 0.1). The
    paper's run (OpenAI's clip package, likely fp16, 77 tokens) differs from ours (HF weights, fp32,
    data.max_text_len=32) in ways that can flip a few rankings, and one flipped ECCV query moves an ECCV
    i2t or t2i score by about 0.08 (1/1,261 or 1/1,332), its mean by half that. Every deviation is listed
    when any fails."""
    off = eccv_caption_paper_deviations(zero_shot_b32)
    report = {k: round(v, 3) for k, v in zero_shot_b32.items()}
    assert not off, f"(got, paper) beyond tolerance: {off}; all metrics: {report}"
