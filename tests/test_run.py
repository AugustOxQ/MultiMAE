import json
import logging
import subprocess
import sys
from datetime import datetime

import pytest
from omegaconf import OmegaConf

from helpers import REPO, compose_cfg
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run, update_run_json

NOW = datetime(2026, 10, 1, 12, 0, 0)


def cfg_for(tmp_path, *overrides):
    return compose_cfg(f"paths.res_dir={tmp_path / 'res'}", "wandb.enabled=false", *overrides)


def test_run_folder_layout_and_lifecycle(tmp_path):
    cfg = cfg_for(tmp_path, "wandb.group=ablation")
    with Run(cfg, now=NOW) as run:
        assert run.path == tmp_path / "res" / "multimae" / "ablation" / "20261001_120000_fusion_concat"
        assert json.loads((run.path / "run.json").read_text())["status"] == "running"
        logging.getLogger("mmae.test").info("hello from the run")
        MetricLogger(run).log({"train/loss": 2.0, "epoch": 1}, step=10)
        MetricLogger(run).log({"val/loss": 1.5, "epoch": 1}, step=20)
        assert not (run.path / "checkpoints").exists()
        ckpt = run.save_checkpoint("best.pt", {"x": 1})
        assert ckpt == run.path / "checkpoints" / "best.pt" and ckpt.exists()
        run.update(results={"best_epoch": 1})
    info = json.loads((run.path / "run.json").read_text())
    assert info["status"] == "completed" and info["results"] == {"best_epoch": 1}
    assert {"commit", "dirty"} <= set(info["git"]) and info["duration_s"] >= 0
    assert OmegaConf.load(run.path / "config.yaml") == OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    lines = [json.loads(line) for line in (run.path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0] == {"step": 10, "metrics": {"train/loss": 2.0, "epoch": 1}}
    assert "hello from the run" in (run.path / "train.log").read_text()
    assert (run.path / "plots" / "curves.png").exists()
    assert not (run.path / "error.txt").exists()


def test_disabled_run_writes_nothing(tmp_path):
    with Run(cfg_for(tmp_path), enabled=False) as run:
        run.log_metrics({"a": 1.0}, step=1)
        run.update(x=1)
        assert run.save_checkpoint("best.pt", {}) is None and run.path is None
    assert not (tmp_path / "res").exists()


def test_same_second_same_name_gets_distinct_folders(tmp_path):
    cfg = cfg_for(tmp_path, "wandb.name=seed")
    with Run(cfg, now=NOW) as first, Run(cfg, now=NOW) as second:
        assert first.path != second.path
        assert second.path.name == "20261001_120000_seed_1"


def test_crash_marks_run_failed_and_reraises(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with Run(cfg_for(tmp_path), now=NOW) as run:
            raise RuntimeError("boom")
    info = json.loads((run.path / "run.json").read_text())
    assert info["status"] == "failed"
    assert "RuntimeError: boom" in (run.path / "error.txt").read_text()


def test_update_run_json_merges(tmp_path):
    with Run(cfg_for(tmp_path), now=NOW) as run:
        pass
    update_run_json(run.path, eval={"test": {"rsum": 1.0}})
    info = json.loads((run.path / "run.json").read_text())
    assert info["eval"] == {"test": {"rsum": 1.0}} and info["status"] == "completed"


def test_list_runs_filters(tmp_path):
    with Run(cfg_for(tmp_path, "wandb.group=a"), now=NOW):
        pass
    with pytest.raises(RuntimeError):
        with Run(cfg_for(tmp_path, "wandb.group=b"), now=NOW):
            raise RuntimeError("x")
    script = [sys.executable, str(REPO / "scripts" / "list_runs.py"), "--root", str(tmp_path / "res")]
    everything = subprocess.run(script, capture_output=True, text=True, check=True).stdout
    assert "completed" in everything and "failed" in everything
    failed = subprocess.run(script + ["--status", "failed"], capture_output=True, text=True, check=True).stdout
    assert "failed" in failed and "completed" not in failed
