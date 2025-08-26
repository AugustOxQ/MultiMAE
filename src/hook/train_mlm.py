import os
from typing import Optional, Any
import math
import time
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.model.mlm import MLM_Transformer
from src.utils import setup_seed, count_parameters
from src.metrics import calculate_mlm_loss


class RandomTokenDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random toy data for demonstration
        tokens = torch.randint(low=5, high=self.vocab_size, size=(self.seq_len,))
        return tokens


def train_mlm(
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 42,
    vocab_size: int = 30522,
    seq_len: int = 128,
    num_workers: int = 4,
    save_dir: Optional[str] = None,
    save_interval: int = 5,
    backbone: Optional[str] = None,
    mask_ratio: float = 0.15,
    model_kwargs: Optional[dict] = None,
    logger: Any = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    setup_seed(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_loader is None or val_loader is None:
        dataset = RandomTokenDataset(
            vocab_size=vocab_size, seq_len=seq_len, num_samples=10000
        )
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    # build model
    kwargs = dict(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        emb_dim=(model_kwargs or {}).get("emb_dim", 256),
        encoder_layer=(model_kwargs or {}).get("encoder_layer", 6),
        encoder_head=(model_kwargs or {}).get("encoder_head", 4),
        decoder_layer=(model_kwargs or {}).get("decoder_layer", 4),
        decoder_head=(model_kwargs or {}).get("decoder_head", 4),
        mask_ratio=mask_ratio,
        backbone=backbone or (model_kwargs or {}).get("backbone", None),
    )
    model = MLM_Transformer(**kwargs).to(device)
    print("Model Parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    print("Starting MLM training")
    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for _, batch_tokens in pbar:
            batch_tokens = batch_tokens.to(device)

            optimizer.zero_grad()
            logits, mask = model(batch_tokens)
            loss = calculate_mlm_loss(logits, labels=batch_tokens, mask=mask)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                }
            )

            global_step += 1
            if logger is not None:
                logger.log_metrics(
                    {
                        "train/step_loss": float(loss.item()),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

        avg_train_loss = sum(epoch_train_losses) / max(1, len(epoch_train_losses))
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            vpbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation {epoch+1}/{epochs}",
            )
            for _, batch_tokens in vpbar:
                batch_tokens = batch_tokens.to(device)
                logits, mask = model(batch_tokens)
                val_loss = calculate_mlm_loss(logits, labels=batch_tokens, mask=mask)
                epoch_val_losses.append(val_loss.item())
                if logger is not None:
                    logger.log_metrics(
                        {"val/step_loss": float(val_loss.item())}, step=global_step
                    )

        avg_val_loss = sum(epoch_val_losses) / max(1, len(epoch_val_losses))
        val_losses.append(avg_val_loss)
        print(
            f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )
        lr_scheduler.step()

        if save_dir and (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"mlm_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at: {save_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "last_lr": lr_scheduler.get_last_lr()[0],
    }
