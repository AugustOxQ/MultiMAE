"""One results folder per training run, adapted from CoSiR's ExperimentManager.

    res/<wandb.project>/<wandb.group or default>/<YYYYMMDD_HHMMSS>_<name>[_n]/
        config.yaml    resolved Hydra config
        run.json       status, timings, git commit, command, host, wandb, results
        metrics.jsonl  one line per logged step
        train.log      main-process console log
        error.txt      traceback, only if the run failed
        checkpoints/   created on first save
        plots/         learning curves drawn at the end

Only the main process writes; other ranks get a disabled Run whose methods do nothing. There is no
shared registry file (parallel runs would race on it); scripts/list_runs.py scans run.json files.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
PLOTTED_PREFIXES = ("train/loss", "val/loss", "val/retrieval/rsum", "val/retrieval/i2t_R1", "val/retrieval/t2i_R1")


def git_info(cwd: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def update_run_json(run_dir: Path, **fields: Any) -> None:
    path = Path(run_dir) / "run.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(fields)
    _write_json(path, info)


class Run:
    def __init__(self, cfg: DictConfig, enabled: bool = True, now: datetime | None = None) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.name = cfg.wandb.name or cfg.model.name
        self.path: Path | None = None
        self._now = now
        self._info: dict[str, Any] = {}
        self._start = 0.0
        self._handler: logging.Handler | None = None

    def _make_dir(self) -> Path:
        stamp = (self._now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        base = Path(self.cfg.paths.res_dir) / self.cfg.wandb.project / (self.cfg.wandb.group or "default")
        base.mkdir(parents=True, exist_ok=True)
        for attempt in range(1000):
            path = base / (f"{stamp}_{self.name}" + (f"_{attempt}" if attempt else ""))
            try:
                path.mkdir()  # atomic: two runs in the same second cannot both get this name
                return path
            except FileExistsError:
                continue
        raise RuntimeError(f"no free run folder name under {base}")

    def __enter__(self) -> "Run":
        if not self.enabled:
            return self
        self.path = self._make_dir()
        OmegaConf.save(self.cfg, self.path / "config.yaml", resolve=True)
        self._start = time.time()
        self._info = {
            "name": self.name,
            "status": "running",
            "model": self.cfg.model.name,
            "group": self.cfg.wandb.group or "default",
            "tags": list(self.cfg.wandb.tags),
            "path": str(self.path.resolve()),
            "created": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "host": socket.gethostname(),
            "num_processes": int(os.environ.get("WORLD_SIZE", "1")),
            "git": git_info(Path.cwd()),
        }
        _write_json(self.path / "run.json", self._info)
        self._handler = logging.FileHandler(self.path / "train.log")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(self._handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        self._info.update(
            status="failed" if exc_type else "completed",
            ended=datetime.now().isoformat(timespec="seconds"),
            duration_s=round(time.time() - self._start, 1),
        )
        if exc_type is not None:
            (self.path / "error.txt").write_text("".join(traceback.format_exception(exc_type, exc, tb)))
        _write_json(self.path / "run.json", self._info)
        try:
            self._plot()
        except Exception:  # plotting must never hide the run's own result
            log.exception("could not draw learning curves")
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
        return False  # never swallow the exception

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        with open(self.path / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "metrics": metrics}, default=float) + "\n")

    def update(self, **fields: Any) -> None:
        if not self.enabled:
            return
        self._info.update(fields)
        _write_json(self.path / "run.json", self._info)

    def save_checkpoint(self, name: str, obj: Any) -> Path | None:
        if not self.enabled:
            return None
        directory = self.path / "checkpoints"
        directory.mkdir(exist_ok=True)
        torch.save(obj, directory / name)
        return directory / name

    def _plot(self) -> None:
        metrics_file = self.path / "metrics.jsonl"
        if not metrics_file.exists():
            return
        series: dict[str, list[tuple[int, float]]] = {}
        for line in metrics_file.read_text().splitlines():
            entry = json.loads(line)
            for key, value in entry["metrics"].items():
                if key.startswith(PLOTTED_PREFIXES) and isinstance(value, (int, float)):
                    series.setdefault(key, []).append((entry["step"], value))
        if not series:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = min(3, len(series))
        rows = -(-len(series) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
        for ax, (key, points) in zip(axes.flat, sorted(series.items())):
            steps, values = zip(*points)
            ax.plot(steps, values, marker="o" if len(points) < 30 else None)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        for ax in list(axes.flat)[len(series):]:
            ax.axis("off")
        fig.tight_layout()
        (self.path / "plots").mkdir(exist_ok=True)
        fig.savefig(self.path / "plots" / "curves.png", dpi=120)
        plt.close(fig)
