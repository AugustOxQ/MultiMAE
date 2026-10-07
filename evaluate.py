"""Retrieval evaluation of a trained run or of the zero-shot pretrained backbone.

  python evaluate.py eval.run_dir=res/multimae/default/20261001_120000_fusion_concat   # a trained run
  python evaluate.py model=fusion_concat eval.split=test                               # zero-shot CLIP

On the full COCO test split, eval.extended_metrics adds ECCV Caption, CxC, COCO 1K/5K and PMRP (mmae.engine.eccv).
"""
import json
import logging
import os
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mmae.data import Collator, build_image_transform, build_retrieval
from mmae.engine.eccv import build_extended_metrics
from mmae.engine.retrieval import encode_retrieval_set, retrieval_metrics
from mmae.engine.vwsd import evaluate_vwsd
from mmae.models import MultiMAE
from mmae.models.backbones import processor_name
from mmae.utils.logging import setup_logging
from mmae.utils.run import update_run_json

# The HF fast tokenizer otherwise warns and can deadlock after fork in DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

log = logging.getLogger("evaluate")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(mixed_precision=cfg.train.precision)
    setup_logging(accelerator.is_main_process)
    split = cfg.eval.split
    run_dir = Path(cfg.eval.run_dir) if cfg.eval.run_dir else None
    if run_dir is not None:
        run_cfg = OmegaConf.load(run_dir / "config.yaml")
        model_cfg, max_text_len = run_cfg.model, run_cfg.data.max_text_len
    else:
        model_cfg, max_text_len = cfg.model, cfg.data.max_text_len
    if set(model_cfg.modalities) != {"image", "text"}:
        raise SystemExit(f"retrieval needs a model with both modalities; {model_cfg.name} has {list(model_cfg.modalities)}")

    # Main process only (all processes hold the same gathered embeddings); built before the model so a
    # misconfiguration fails fast.
    extended = build_extended_metrics(cfg, split, two_modalities=True) if accelerator.is_main_process else None

    model = MultiMAE(model_cfg, max_text_len=max_text_len)
    if run_dir is not None:
        checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        log.info("loaded %s (epoch %s)", run_dir / "checkpoints" / "best.pt", checkpoint.get("epoch"))
    else:
        log.info("zero-shot evaluation of %s", model_cfg.backbone.pretrained)

    data, processor = cfg.data, processor_name(model_cfg.backbone)
    dataset = build_retrieval(
        data, split, build_image_transform(processor), data.limit_val if split == "val" else data.limit_test,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers,
        collate_fn=Collator(processor, max_text_len).retrieval,
    )
    model, loader = accelerator.prepare(model, loader)
    images, captions = encode_retrieval_set(model, loader, accelerator)
    metrics = retrieval_metrics(images, captions)
    if accelerator.is_main_process:
        if extended is not None:
            metrics.update(extended(images, captions))
        if cfg.eval.get("vwsd_dir"):
            metrics.update(evaluate_vwsd(
                accelerator.unwrap_model(model), cfg.eval.vwsd_dir, build_image_transform(processor),
                Collator(processor, max_text_len), lang=cfg.eval.vwsd_lang, prompt=cfg.eval.vwsd_prompt,
                batch_size=cfg.train.eval_batch_size, num_workers=cfg.train.num_workers, device=accelerator.device,
            ))
        log.info("%s retrieval on %d images: %s", split, len(dataset),
                 ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
        if run_dir is not None:
            info = json.loads((run_dir / "run.json").read_text())
            update_run_json(run_dir, eval={**info.get("eval", {}), split: metrics})
        if cfg.eval.output:
            output = Path(cfg.eval.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(metrics, indent=2))
    accelerator.end_training()


if __name__ == "__main__":
    main()
