import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from src.hook import train_mlm
from src.utils import SimpleWandbLogger, setup_seed


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_split, tokenizer, seq_len: int):
        self.ds = hf_dataset_split
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        text = (
            self.ds[idx]["text"]
            if "text" in self.ds.column_names
            else str(self.ds[idx])
        )
        toks = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.seq_len,
            return_tensors="pt",
        )
        return toks["input_ids"].squeeze(0)


@hydra.main(version_base=None, config_path="configs", config_name="mlm_config")
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
        wandb.define_metric("mlm/train/epoch_loss", step_metric="epoch")
        wandb.define_metric("mlm/val/epoch_loss", step_metric="epoch")
        wandb_logger = SimpleWandbLogger()

    # data: use huggingface datasets (e.g., wikitext-2-raw-v1)
    from transformers import AutoTokenizer

    tok_name = cfg.text.backbone if cfg.text.backbone else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    seq_len = cfg.text.seq_len
    train_split = HFDatasetWrapper(ds["train"], tokenizer, seq_len)
    val_split = HFDatasetWrapper(ds["validation"], tokenizer, seq_len)

    train_loader = DataLoader(
        train_split,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    results = train_mlm(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        batch_size=cfg.train.batch_size,
        weight_decay=cfg.train.weight_decay,
        seed=cfg.train.seed,
        vocab_size=cfg.text.vocab_size,
        seq_len=cfg.text.seq_len,
        num_workers=cfg.train.num_workers,
        save_dir=None,  # Note during testing, we do not save model checkpoints
        save_interval=cfg.train.save_interval,
        backbone=cfg.text.backbone,
        mask_ratio=cfg.text.mask_ratio,
        model_kwargs={
            "emb_dim": cfg.text.emb_dim,
            "encoder_layer": cfg.text.encoder_layer,
            "encoder_head": cfg.text.encoder_head,
            "decoder_layer": cfg.text.decoder_layer,
            "decoder_head": cfg.text.decoder_head,
        },
        logger=wandb_logger,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # epoch-wise logs
    if wandb_logger is not None:
        for epoch, (tr, va) in enumerate(
            zip(results["train_losses"], results["val_losses"]), start=1
        ):
            wandb_logger.log_metrics(
                {"mlm/train/epoch_loss": tr, "mlm/val/epoch_loss": va, "epoch": epoch}
            )

    if cfg.get("wandb") and cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
