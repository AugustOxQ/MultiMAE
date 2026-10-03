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
        f"paths.res_dir={tmp_path / 'res'}", "eval.extended_metrics=false", *overrides,
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


def test_extended_metrics_reach_the_test_split_only(tmp_path, fake_coco, accelerator, monkeypatch):
    """With the gate passing, Trainer.evaluate("test") adds the extended metrics under test/, computed on
    the gathered test embeddings; val never gets them."""
    import mmae.engine.trainer as trainer_module

    calls = []

    def fake_build(cfg, split, two_modalities):
        assert split == "test" and two_modalities

        def extended(images, captions):
            calls.append((images.shape, captions.shape))
            return {"eccv/map_at_r": 12.5, "pmrp/i2t": 40.0}

        return extended

    monkeypatch.setattr(trainer_module, "build_extended_metrics", fake_build)
    cfg = tiny_cfg(tmp_path, fake_coco, "eval.extended_metrics=true")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        val, test = trainer.evaluate("val"), trainer.evaluate("test")
    assert not any(k.startswith(("val/eccv", "val/pmrp")) for k in val)
    assert test["test/eccv/map_at_r"] == 12.5 and test["test/pmrp/i2t"] == 40.0
    assert "test/retrieval/rsum" in test
    assert len(calls) == 1 and calls[0][0][0] == 6 and calls[0][1][:2] == (6, 5)  # 6 test images, 5 captions each


@pytest.mark.parametrize("limit_test", ["null", "3"])
def test_extended_metrics_skipped_on_fake_coco(tmp_path, fake_coco, accelerator, limit_test):
    """Switched on, they still skip the fake COCO's 6-image test split, and any limit_test."""
    cfg = tiny_cfg(tmp_path, fake_coco, "eval.extended_metrics=true", f"data.limit_test={limit_test}")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        test = trainer.evaluate("test")
    assert trainer.extended_metrics is None
    assert "test/retrieval/rsum" in test and not any(k.startswith(("test/eccv", "test/pmrp", "test/coco1k")) for k in test)


def _snapshot(params):
    return [p.detach().clone() for p in params]


def _same(params, snapshot):
    return all(torch.equal(p, s) for p, s in zip(params, snapshot))


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_frozen_vision_warmup(tmp_path, fake_coco, grad_accum):
    # train.py builds the Accelerator with gradient_accumulation_steps=train.grad_accum; the shared fixture does not
    AcceleratorState._reset_state(True)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=grad_accum)
    cfg = tiny_cfg(tmp_path, fake_coco, "model=contrastive", "train.epochs=2", "train.freeze_vision_epochs=1",
                   f"train.grad_accum={grad_accum}")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        model = accelerator.unwrap_model(trainer.model)
        vision, text = list(model.vision.pretrained_parameters()), list(model.text.pretrained_parameters())
        v0, t0 = _snapshot(vision), _snapshot(text)
        trainer.train_epoch(1)
        assert _same(vision, v0), "vision tower moved during the frozen epoch"
        assert not _same(text, t0), "text tower did not train"
        trainer.train_epoch(2)
        assert not _same(vision, v0), "vision tower did not train after the frozen epoch"
    AcceleratorState._reset_state(True)


def test_split_groups_log_per_tower_lrs(tmp_path, fake_coco, accelerator, monkeypatch):
    cfg = tiny_cfg(tmp_path, fake_coco, "model=contrastive", "train.epochs=2", "train.freeze_vision_epochs=1",
                   "train.log_every=1")
    with Run(cfg, enabled=False) as run:
        logged = []
        metric_logger = MetricLogger(run)
        monkeypatch.setattr(metric_logger, "log", lambda metrics, step=None: logged.append(dict(metrics)))
        trainer = Trainer(cfg, accelerator, run, metric_logger)
        trainer.train_epoch(1)
        trainer.train_epoch(2)
    train_logs = [m for m in logged if "train/lr" in m or "train/lr_vision" in m or "train/lr_backbone" in m]
    # the lr is read after the scheduler step, so the first log of epoch 1 still shows the frozen lr
    first, second = ([m for m in train_logs if m["epoch"] == e][i] for e, i in ((1, 0), (2, 0)))
    assert "train/lr_backbone" not in first
    assert first["train/lr_vision"] == 0 and first["train/lr_text"] > 0
    assert second["train/lr_vision"] > 0


def test_default_schedule_keeps_the_legacy_groups(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco, "model=fusion_concat")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        names = sorted(g["name"] for g in trainer.optimizer.param_groups)
    assert names == ["backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"]


def _first_epoch_order(tmp_path, fake_coco, accelerator, *overrides):
    cfg = tiny_cfg(tmp_path, fake_coco, *overrides)
    torch.manual_seed(cfg.seed)  # as train.py's set_seed
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        return torch.cat([batch["input_ids"] for batch in trainer.train_loader])


def test_seeded_sampler_gives_every_model_the_same_order(tmp_path, fake_coco, accelerator):
    on = ("train.seeded_sampler=true", "train.batch_size=2")
    a = _first_epoch_order(tmp_path / "a", fake_coco, accelerator, "model=contrastive", *on)
    b = _first_epoch_order(tmp_path / "b", fake_coco, accelerator, "model=fusion_multilearner", *on)
    c = _first_epoch_order(tmp_path / "c", fake_coco, accelerator, "model=contrastive", *on, "seed=7")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
