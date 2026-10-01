import json

import pytest

from helpers import MODEL_NAMES, run_train


def single_run_dir(cwd):
    runs = sorted((cwd / "res").glob("multimae/default/*"))
    assert len(runs) == 1, runs
    return runs[0]


def check_run(run_dir, two_modalities: bool):
    info = json.loads((run_dir / "run.json").read_text())
    assert info["status"] == "completed"
    assert info["results"]["best_epoch"] == 1
    assert "test/loss" in info["results"]["test"]
    assert ("test/retrieval/rsum" in info["results"]["test"]) == two_modalities
    for name in ("config.yaml", "metrics.jsonl", "train.log", "plots/curves.png"):
        assert (run_dir / name).exists(), name
    return info


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_debug_run_on_cpu(tmp_path, fake_coco, name):
    result = run_train(tmp_path, fake_coco, f"model={name}", "train.save=best")
    assert result.returncode == 0, result.stderr[-5000:]
    run_dir = single_run_dir(tmp_path)
    check_run(run_dir, two_modalities=name.startswith("fusion"))
    assert (run_dir / "checkpoints" / "best.pt").exists()
    # Hydra and wandb write nothing outside the run folder
    assert not (tmp_path / "outputs").exists() and not (tmp_path / ".hydra").exists()
    assert not (tmp_path / "wandb").exists()


def test_train_two_cpu_processes(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "model=fusion_concat", nproc=2)
    assert result.returncode == 0, result.stderr[-5000:]
    info = check_run(single_run_dir(tmp_path), two_modalities=True)  # one folder, not one per rank
    assert info["num_processes"] == 2


@pytest.mark.slow
@pytest.mark.parametrize("name", ["fusion_concat", "fusion_multilearner"])
def test_train_debug_run_on_gpu_real_clip(tmp_path, name):
    import os
    import subprocess
    import sys

    from helpers import REPO

    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_USE_CPU")}
    cmd = [sys.executable, str(REPO / "train.py"), f"model={name}", "train=debug", f"paths.res_dir={tmp_path / 'res'}"]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-5000:]
    check_run(single_run_dir(tmp_path), two_modalities=True)
