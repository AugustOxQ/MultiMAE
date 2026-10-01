import hashlib
import json
import math

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from helpers import compose_cfg
from mmae.engine.trainer import EarlyStopper, Trainer, warmup_cosine
from mmae.losses import MAX_LOGIT_SCALE
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run


@pytest.fixture
def accelerator():
    acc = Accelerator(cpu=True)
    yield acc
    AcceleratorState._reset_state(True)


def tiny_cfg(tmp_path, fake_coco, *overrides):
    images_dir, annotations_dir = fake_coco
    return compose_cfg(
        f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
        "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
        f"paths.res_dir={tmp_path / 'res'}", *overrides,
    )


def weights_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_warmup_cosine():
    assert warmup_cosine(0, 10, 110) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 110) == pytest.approx(1.0)
    assert warmup_cosine(10, 10, 110) == pytest.approx(1.0)
    assert warmup_cosine(60, 10, 110) == pytest.approx(0.5)
    assert warmup_cosine(110, 10, 110) == pytest.approx(0.0, abs=1e-12)
    assert warmup_cosine(0, 0, 100) == pytest.approx(1.0)


def test_early_stopper():
    stop = EarlyStopper("max", patience=2, min_delta=0.5)
    assert stop.update(10.0) and stop.best == 10.0
    assert not stop.update(10.4) and not stop.should_stop  # within min_delta
    assert stop.update(11.0)
    assert not stop.update(9.0) and not stop.update(9.0) and stop.should_stop
    low = EarlyStopper("min", patience=1, min_delta=0.0)
    assert low.update(3.0) and low.update(2.0) and not low.update(2.5) and low.should_stop
    with pytest.raises(ValueError):
        EarlyStopper("best", 1, 0.0)


def test_fit_restores_best_weights_before_test(tmp_path, fake_coco, accelerator, monkeypatch):
    cfg = tiny_cfg(tmp_path, fake_coco, "train.epochs=3")
    scripted = iter([10.0, 30.0, 20.0])  # best at epoch 2
    hashes: dict[str, str] = {}

    def fake_evaluate(self, split):
        current = weights_hash(self.accelerator.unwrap_model(self.model))
        if split == "val":
            hashes[f"val{sum(k.startswith('val') for k in hashes) + 1}"] = current
            return {"val/loss": 1.0, "val/retrieval/rsum": next(scripted)}
        hashes["test"] = current
        return {"test/loss": 1.0, "test/retrieval/rsum": 0.0}

    monkeypatch.setattr(Trainer, "evaluate", fake_evaluate)
    with Run(cfg) as run:
        results = Trainer(cfg, accelerator, run, MetricLogger(run)).fit()
    assert results["best_epoch"] == 2
    assert hashes["test"] == hashes["val2"] != hashes["val3"]


def test_evaluate_reports_losses_and_retrieval(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco)
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        first, second = trainer.evaluate("val"), trainer.evaluate("val")
    assert {"val/loss", "val/loss_mae", "val/loss_mlm", "val/loss_contrastive", "val/retrieval/rsum"} <= set(first)
    assert first == second  # masks are reseeded for every evaluation
    assert all(math.isfinite(v) for v in first.values())
    assert trainer.model.training


def test_logit_scale_is_clamped_after_each_step(tmp_path, fake_coco, accelerator):
    """The parameter is clamped to [0, log 100] after an optimizer step (open_clip), so the logged
    train/logit_scale is the scale the loss used, never above 100."""
    cfg = tiny_cfg(tmp_path, fake_coco)
    with Run(cfg) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        logit_scale = trainer.accelerator.unwrap_model(trainer.model).logit_scale
        with torch.no_grad():
            logit_scale.fill_(6.0)  # exp(6) = 403
        trainer.train_epoch(1)
        path = run.path / "metrics.jsonl"
    assert trainer.global_step > 0
    assert logit_scale <= torch.tensor(math.log(MAX_LOGIT_SCALE), dtype=logit_scale.dtype), logit_scale.item()
    logged = [json.loads(line)["metrics"]["train/logit_scale"] for line in path.read_text().splitlines()]
    assert logged and all(value <= MAX_LOGIT_SCALE + 1e-3 for value in logged), logged


def test_too_small_training_set_fails_clearly(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco, "data.limit_train=4")
    with Run(cfg, enabled=False) as run, pytest.raises(ValueError, match="no full batch"):
        Trainer(cfg, accelerator, run, MetricLogger(run))
