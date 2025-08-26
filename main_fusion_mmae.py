import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.hook import train_fusionmmae
from src.utils import SimpleWandbLogger, setup_seed


@hydra.main(version_base=None, config_path="configs", config_name="fusion_mmae_config")
def main(cfg: DictConfig) -> None:
    print("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    setup_seed(cfg.train.seed)

    wandb_logger = None
    if cfg.get("wandb") and cfg.wandb.enabled:
        if getattr(cfg.wandb, "mode", "") == "disabled":
            os.environ["WANDB_MODE"] = "disabled"
        elif getattr(cfg.wandb, "mode", "") == "offline":
            os.environ["WANDB_MODE"] = "offline"
        run_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=getattr(cfg.wandb, "project", None),
            entity=getattr(cfg.wandb, "entity", None),
            name=(cfg.wandb.name if getattr(cfg.wandb, "name", "") else None),
            config=run_config,
            tags=(list(cfg.wandb.tags) if getattr(cfg.wandb, "tags", None) else None),
            notes=(cfg.wandb.notes if getattr(cfg.wandb, "notes", None) else None),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/epoch_total_loss", step_metric="epoch")
        wandb.define_metric("train/epoch_mae_loss", step_metric="epoch")
        wandb.define_metric("train/epoch_contrastive_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_total_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_mae_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_mlm_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_contrastive_loss", step_metric="epoch")
        wandb.define_metric("test/epoch_total_loss", step_metric="epoch")
        wandb.define_metric("test/epoch_mae_loss", step_metric="epoch")
        wandb.define_metric("test/epoch_mlm_loss", step_metric="epoch")
        wandb.define_metric("test/epoch_contrastive_loss", step_metric="epoch")
        # Early stopping metrics
        wandb.define_metric("train/best_val_loss", step_metric="epoch")
        wandb.define_metric("train/early_stopped", step_metric="epoch")
        wandb_logger = SimpleWandbLogger()

    results = train_fusionmmae(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        batch_size=cfg.train.batch_size,
        weight_decay=cfg.train.weight_decay,
        seed=cfg.train.seed,
        image_size=cfg.model.image_size,
        data_root=cfg.train.data_root,
        num_workers=cfg.train.num_workers,
        temperature=cfg.loss.temperature,
        backbone_vision=cfg.model.backbone,
        text_backbone=cfg.text.backbone,
        text_max_len=cfg.text.max_len,
        proj_dim=cfg.model.proj_dim,
        fusion_method=cfg.model.fusion_method,
        mae_weight=cfg.loss.mae_weight,
        mlm_weight=cfg.loss.mlm_weight,
        contrastive_weight=cfg.loss.contrastive_weight,
        save_dir=None,  # Note during testing, we do not save model checkpoints
        save_interval=cfg.train.save_interval,
        logger=wandb_logger,
        # Early stopping parameters
        patience=cfg.train.patience,
        min_delta=cfg.train.min_delta,
    )

    if wandb_logger is not None:
        for epoch in range(1, len(results["train_total_losses"]) + 1):
            wandb_logger.log_metrics(
                {
                    "train/epoch_total_loss": results["train_total_losses"][epoch - 1],
                    "train/epoch_mae_loss": results["train_mae_losses"][epoch - 1],
                    "train/epoch_mlm_loss": results["train_mlm_losses"][epoch - 1],
                    "train/epoch_contrastive_loss": results["train_contrastive_losses"][
                        epoch - 1
                    ],
                    "val/epoch_total_loss": results["val_total_losses"][epoch - 1],
                    "val/epoch_mae_loss": results["val_mae_losses"][epoch - 1],
                    "val/epoch_mlm_loss": results["val_mlm_losses"][epoch - 1],
                    "val/epoch_contrastive_loss": results["val_contrastive_losses"][
                        epoch - 1
                    ],
                    "epoch": epoch,
                }
            )

        # Log early stopping results (single values, not epoch-based)
        wandb_logger.log_metrics(
            {
                "train/best_val_loss": results["best_val_loss"],
                "train/early_stopped": results["early_stopped"],
                "train/final_epoch": results["final_epoch"],
            }
        )

        # Log test results (single values, not epoch-based)
        wandb_logger.log_metrics(
            {
                "test/total_loss": results["test_total_loss"],
                "test/mae_loss": results["test_mae_loss"],
                "test/mlm_loss": results["test_mlm_loss"],
                "test/contrastive_loss": results["test_contrastive_loss"],
            }
        )

    if cfg.get("wandb") and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
