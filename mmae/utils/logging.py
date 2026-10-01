"""Console logging, metric logging (run folder + wandb) and wandb setup."""
from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mmae.utils.run import Run

log = logging.getLogger(__name__)
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "PIL", "matplotlib")


def setup_logging(is_main: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class MetricLogger:
    """Writes metrics to the run folder and wandb; prints val/test metrics to the console."""

    def __init__(self, run: Run, wandb_run: Any = None) -> None:
        self.run = run
        self.wandb_run = wandb_run

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.run.log_metrics(metrics, step)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        if any(key.startswith(("val/", "test/")) for key in metrics):
            shown = ", ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float)))
            log.info("step %d: %s", step, shown)


def init_wandb(cfg: DictConfig, run: Run):
    """Start a wandb run (files under the run folder) and record its id/url in run.json; None if disabled."""
    if not cfg.wandb.enabled:
        return None
    import wandb

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        group=cfg.wandb.group or None,
        name=run.name,
        tags=list(cfg.wandb.tags) or None,
        notes=cfg.wandb.notes or None,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run.path) if run.path else None,
    )
    run.update(wandb={"id": wandb_run.id, "url": wandb_run.url})
    return wandb_run
