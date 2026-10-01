"""Train a MultiMAE model.

  python train.py model=fusion_concat                                 # one GPU
  accelerate launch --num_processes 4 --multi_gpu train.py ...        # one node, several GPUs
  python train.py train=debug                                         # tiny smoke run, wandb off
"""
import os

import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig

from mmae.engine.trainer import Trainer
from mmae.utils.logging import MetricLogger, init_wandb, setup_logging
from mmae.utils.run import Run

# The HF fast tokenizer otherwise warns and can deadlock after fork in DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        mixed_precision=cfg.train.precision, gradient_accumulation_steps=cfg.train.grad_accum
    )
    setup_logging(accelerator.is_main_process)
    set_seed(cfg.seed, device_specific=True)  # different masks per process; DDP syncs the initial weights
    with Run(cfg, enabled=accelerator.is_main_process) as run:
        wandb_run = init_wandb(cfg, run) if accelerator.is_main_process else None
        try:
            Trainer(cfg, accelerator, run, MetricLogger(run, wandb_run)).fit()
        finally:
            if wandb_run is not None:
                wandb_run.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
