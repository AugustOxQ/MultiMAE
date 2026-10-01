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


def test_evaluate_rejects_single_modality(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "model=image_mae", script="evaluate.py")
    assert result.returncode != 0 and "both modalities" in result.stderr


@pytest.mark.slow
def test_zero_shot_clip_b32_matches_published(tmp_path):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_USE_CPU")}
    out = tmp_path / "zero_shot.json"
    cmd = [sys.executable, str(REPO / "evaluate.py"), "eval.split=test", "train.eval_batch_size=256", f"eval.output={out}"]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-5000:]
    metrics = json.loads(out.read_text())
    # OpenAI CLIP ViT-B/32 zero-shot, COCO 5k test: i2t R@1 50.1, t2i R@1 30.4 (published)
    assert abs(metrics["i2t_R1"] - 50.1) < 1.5, metrics
    assert abs(metrics["t2i_R1"] - 30.4) < 1.5, metrics
