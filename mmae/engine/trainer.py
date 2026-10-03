"""Training loop: per-epoch validation with retrieval, early stopping, best-weight restore, final test."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mmae.data import CocoPairs, CocoRetrieval, Collator, build_image_transform
from mmae.engine.eccv import build_extended_metrics
from mmae.engine.retrieval import encode_retrieval_set, retrieval_metrics
from mmae.losses import MAX_LOGIT_SCALE
from mmae.models import MultiMAE
from mmae.models.backbones import processor_name
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run

log = logging.getLogger(__name__)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1 over `warmup_steps`, then cosine decay to 0 at `total_steps`."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"monitor mode must be 'max' or 'min', got {mode!r}")
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Record a validation value; True if it is a new best (by more than min_delta)."""
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@dataclass
class EvalLoaders:
    pairs: DataLoader
    retrieval: DataLoader | None


class Trainer:
    def __init__(self, cfg: DictConfig, accelerator: Accelerator, run: Run, metric_logger: MetricLogger) -> None:
        self.cfg, self.accelerator, self.run, self.metric_logger = cfg, accelerator, run, metric_logger
        tcfg, dcfg, mcfg = cfg.train, cfg.data, cfg.model
        self.two_modalities = set(mcfg.modalities) == {"image", "text"}
        processor = processor_name(mcfg.backbone)
        self.transform = build_image_transform(processor)
        content_words = str(mcfg.masking.get("text_mode", "random")) == "content"
        self.collator = Collator(processor, dcfg.max_text_len, content_words=content_words)

        model = MultiMAE(mcfg, max_text_len=dcfg.max_text_len)
        train_set = CocoPairs(dcfg.images_dir, dcfg.annotations_dir, "train", self.transform, dcfg.limit_train)
        generator = torch.Generator().manual_seed(int(cfg.seed)) if tcfg.get("seeded_sampler", False) else None
        train_loader = self._loader(train_set, tcfg.batch_size, shuffle=True, drop_last=True,
                                    collate=self.collator.pairs, generator=generator)
        freeze_epochs = int(tcfg.get("freeze_vision_epochs", 0))
        optimizer = torch.optim.AdamW(model.param_groups(
            tcfg.lr, tcfg.lr_backbone, tcfg.weight_decay, lr_text=tcfg.get("lr_text"),
            lr_vision=tcfg.get("lr_vision"), layer_decay=float(tcfg.get("layer_decay", 1.0)),
            split_towers=freeze_epochs > 0,
        ))
        self.model, self.optimizer, self.train_loader = accelerator.prepare(model, optimizer, train_loader)
        if len(self.train_loader) == 0:
            raise ValueError(
                f"training set of {len(train_set)} items yields no full batch of {tcfg.batch_size} "
                f"per process ({accelerator.num_processes} processes)"
            )
        steps_per_epoch = math.ceil(len(self.train_loader) / tcfg.grad_accum)
        total_steps = steps_per_epoch * tcfg.epochs

        def schedule(step: int) -> float:
            return warmup_cosine(step, tcfg.warmup_steps, total_steps)

        lambdas = schedule
        if freeze_epochs > 0:  # R2: vision lr 0 for the first freeze_epochs epochs, counted in optimizer steps
            freeze_steps = steps_per_epoch * freeze_epochs

            def frozen(step: int) -> float:
                return 0.0 if step < freeze_steps else schedule(step)

            lambdas = [frozen if group.get("tower") == "vision" else schedule for group in optimizer.param_groups]
        # Built on the raw optimizer and stepped by hand once per optimizer step, so it does not depend on
        # accelerate's scheduler stepping rules.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
        self.eval_loaders: dict[str, EvalLoaders] = {}
        self.global_step = 0
        # ECCV Caption, CxC, COCO 1K and PMRP on the test split, on the main process only: every process holds
        # the same gathered embeddings, nothing after the test is a collective, and the PM ground truth takes
        # about 1 GB of RAM per process. Built here so a misconfigured run fails before training.
        self.extended_metrics = (
            build_extended_metrics(cfg, "test", self.two_modalities) if accelerator.is_main_process else None
        )

    def _loader(self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool, collate,
                generator: torch.Generator | None = None) -> DataLoader:
        workers = int(self.cfg.train.num_workers)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate,
            num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
            generator=generator,
        )

    def _eval_loaders(self, split: str) -> EvalLoaders:
        if split not in self.eval_loaders:
            d, batch = self.cfg.data, self.cfg.train.eval_batch_size
            limit = d.limit_val if split == "val" else d.limit_test
            pairs = self._loader(
                CocoPairs(d.images_dir, d.annotations_dir, split, self.transform, limit),
                batch, shuffle=False, drop_last=False, collate=self.collator.pairs,
            )
            retrieval = None
            if self.two_modalities:
                retrieval = self._loader(
                    CocoRetrieval(d.images_dir, d.annotations_dir, split, self.transform, limit),
                    batch, shuffle=False, drop_last=False, collate=self.collator.retrieval,
                )
                pairs, retrieval = self.accelerator.prepare(pairs, retrieval)
            else:
                pairs = self.accelerator.prepare(pairs)
            self.eval_loaders[split] = EvalLoaders(pairs, retrieval)
        return self.eval_loaders[split]

    def _reduce_means(self, sums: dict[str, float], count: int) -> dict[str, float]:
        keys = sorted(sums)
        values = torch.tensor([sums[k] / max(count, 1) for k in keys], device=self.accelerator.device)
        return dict(zip(keys, self.accelerator.reduce(values, reduction="mean").tolist()))

    def _log_train(self, sums: dict[str, float], count: int, epoch: int) -> None:
        metrics: dict[str, Any] = {f"train/{k}": v for k, v in self._reduce_means(sums, count).items()}
        groups = self.optimizer.param_groups
        if any(group.get("tower") is not None for group in groups):
            # split groups: each tower's lr is read from its top layer (the largest layer index)
            top: dict[str, tuple[int, float]] = {}
            for group in groups:
                tower = group.get("tower")
                if tower is not None:
                    layer = int(group["name"].split("_layer")[1][:2])
                    if tower not in top or layer > top[tower][0]:
                        top[tower] = (layer, group["lr"])
            for tower, (_, lr) in top.items():
                metrics[f"train/lr_{tower}"] = lr
            for group in groups:
                if group["name"].startswith("head"):
                    metrics.setdefault("train/lr", group["lr"])
        else:
            for group in groups:
                key = "train/lr_backbone" if group["name"].startswith("backbone") else "train/lr"
                metrics.setdefault(key, group["lr"])
        logit_scale = self.accelerator.unwrap_model(self.model).logit_scale
        if logit_scale is not None:
            metrics["train/logit_scale"] = logit_scale.exp().item()
        metrics["epoch"] = epoch
        self.metric_logger.log(metrics, step=self.global_step)

    def _clamp_logit_scale(self) -> None:
        """Keep exp(logit_scale) in [1, MAX_LOGIT_SCALE] by clamping the parameter after an optimizer step
        (open_clip's approach), so its gradient is never cut off at the bound."""
        logit_scale = self.accelerator.unwrap_model(self.model).logit_scale
        if logit_scale is not None:
            with torch.no_grad():
                logit_scale.clamp_(0, math.log(MAX_LOGIT_SCALE))

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        tcfg = self.cfg.train
        sums: dict[str, float] = {}
        count = 0
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        for batch in progress:
            with self.accelerator.accumulate(self.model):
                out = self.model(batch)
                self.accelerator.backward(out["loss"])
                if self.accelerator.sync_gradients and tcfg.grad_clip:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.accelerator.sync_gradients:
                    self._clamp_logit_scale()
            for key, value in out.items():
                sums[key] = sums.get(key, 0.0) + value.detach().float().item()
            count += 1
            if self.accelerator.sync_gradients:
                self.scheduler.step()
                self.global_step += 1
                if self.global_step % tcfg.log_every == 0:
                    self._log_train(sums, count, epoch)
                    sums, count = {}, 0
        if count:
            self._log_train(sums, count, epoch)

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        """Mean losses on the split's pairs (masks reseeded so every call sees the same masks) and, for
        two-modality models, retrieval metrics. Identical on every process, except the extended test metrics
        (test/eccv/..., test/cxc/..., test/coco1k/..., test/coco5k/..., test/pmrp...), main process only."""
        loaders = self._eval_loaders(split)
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        cuda = [self.accelerator.device] if self.accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(self.cfg.seed + self.accelerator.process_index)
            for batch in loaders.pairs:
                for key, value in self.model(batch).items():
                    sums[key] = sums.get(key, 0.0) + value.float().item()
                count += 1
        metrics = {f"{split}/{k}": v for k, v in self._reduce_means(sums, count).items()}
        if loaders.retrieval is not None:
            images, captions = encode_retrieval_set(self.model, loaders.retrieval, self.accelerator)
            metrics.update({f"{split}/retrieval/{k}": v for k, v in retrieval_metrics(images, captions).items()})
            if split == "test" and self.extended_metrics is not None:
                metrics.update({f"{split}/{k}": v for k, v in self.extended_metrics(images, captions).items()})
        self.model.train()
        return metrics

    def _save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.cfg.train.save != "best":
            return
        state = self.accelerator.unwrap_model(self.model).state_dict()
        self.run.save_checkpoint(
            "best.pt",
            {"model": state, "config": OmegaConf.to_container(self.cfg, resolve=True), "epoch": epoch, "metrics": metrics},
        )

    def fit(self) -> dict[str, Any]:
        tcfg, monitor = self.cfg.train, self.cfg.model.monitor
        stopper = EarlyStopper(monitor.mode, tcfg.patience, tcfg.min_delta)
        best_state, best_epoch, best_metrics = None, None, {}
        for epoch in range(1, tcfg.epochs + 1):
            self.train_epoch(epoch)
            if epoch % tcfg.eval_every != 0 and epoch != tcfg.epochs:
                continue
            metrics = self.evaluate("val")
            self.metric_logger.log({**metrics, "epoch": epoch}, step=self.global_step)
            if monitor.metric not in metrics:
                raise KeyError(f"monitor metric {monitor.metric!r} not in {sorted(metrics)}")
            if stopper.update(metrics[monitor.metric]):
                best_epoch, best_metrics = epoch, metrics
                state = self.accelerator.unwrap_model(self.model).state_dict()
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
                self._save_best(epoch, metrics)
                log.info("epoch %d: new best %s = %.4f", epoch, monitor.metric, metrics[monitor.metric])
            elif stopper.should_stop:
                log.info("early stopping after epoch %d", epoch)
                break
        if best_state is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict(best_state)
            log.info("restored the best weights (epoch %s) for the test", best_epoch)
        test = self.evaluate("test")
        self.metric_logger.log({**test, "epoch": best_epoch}, step=self.global_step)
        results = {"best_epoch": best_epoch, "best_val": best_metrics, "test": test}
        self.run.update(results=results)
        return results
