import os
import hydra
from omegaconf import DictConfig, OmegaConf

from src.hook import train_mae
from src.utils import SimpleWandbLogger
import wandb


@hydra.main(version_base=None, config_path="configs", config_name="mae_config")
def main(cfg: DictConfig) -> None:
    print("Loaded config:\n" + OmegaConf.to_yaml(cfg))
    # init wandb from hydra config
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
        # define epoch axis to avoid conflict with global step from step-wise logs
        wandb.define_metric("epoch")
        wandb.define_metric("train/epoch_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_loss", step_metric="epoch")
        wandb_logger = SimpleWandbLogger()

    model_cfg = cfg.model
    # auto-map backbone -> default hyperparameters if not explicitly set
    auto_map = {
        "vit_tiny_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "emb_dim": 192,
            "encoder_layer": 12,
            "encoder_head": 3,
        },
        "vit_small_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "emb_dim": 384,
            "encoder_layer": 12,
            "encoder_head": 6,
        },
        "vit_base_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "emb_dim": 768,
            "encoder_layer": 12,
            "encoder_head": 12,
        },
        "vit_large_patch16_224": {
            "image_size": 224,
            "patch_size": 16,
            "emb_dim": 1024,
            "encoder_layer": 24,
            "encoder_head": 16,
        },
    }

    if model_cfg.backbone and model_cfg.backbone in auto_map:
        defaults = auto_map[model_cfg.backbone]
        print(
            f"Using {model_cfg.backbone} backbone with default hyperparameters: {defaults}"
        )
        # only fill values that user did not override explicitly
        for k, v in defaults.items():
            if k not in model_cfg or model_cfg[k] in (None, "", 0):
                model_cfg[k] = v

    # train settings
    train_cfg = cfg.train
    results = train_mae(
        epochs=train_cfg.epochs,
        lr=train_cfg.lr,
        batch_size=train_cfg.batch_size,
        weight_decay=train_cfg.weight_decay,
        seed=train_cfg.seed,
        mask_ratio=model_cfg.mask_ratio,
        save_dir=train_cfg.save_dir,
        data_root=train_cfg.data_root,
        num_workers=train_cfg.num_workers,
        model_kwargs={
            "image_size": model_cfg.image_size,
            "patch_size": model_cfg.patch_size,
            "emb_dim": model_cfg.emb_dim,
            "encoder_layer": model_cfg.encoder_layer,
            "encoder_head": model_cfg.encoder_head,
            "decoder_layer": model_cfg.decoder_layer,
            "decoder_head": model_cfg.decoder_head,
            "mask_ratio": model_cfg.mask_ratio,
            "backbone": model_cfg.backbone,
        },
        logger=wandb_logger,
    )

    # log epoch-wise metrics to wandb
    train_losses = results.get("train_losses", [])
    val_losses = results.get("val_losses", [])
    for epoch, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
        # log without explicit step; use epoch axis defined above
        wandb_logger.log_metrics(
            {
                "train/epoch_loss": tr,
                "val/epoch_loss": va,
                "epoch": epoch,
            }
        )

    last_lr = results.get("last_lr", None)
    best_val = min(val_losses) if len(val_losses) > 0 else None
    if wandb_logger is not None:
        wandb_logger.log_metrics(
            {
                "training/last_lr": last_lr,
                "training/best_val_loss": best_val,
                "epoch": len(train_losses),
            }
        )

    print("Training finished. Last LR:", last_lr)
    if cfg.get("wandb") and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
