import os
from typing import Optional
import math
import time
import torch
import torchvision
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from typing import Any

from src.model import MAE_ViT
from src.utils import setup_seed, count_parameters
from src.metrics import calculate_mae_loss


def train_mae(
    epochs: int = 120,
    lr: float = 1e-4,
    batch_size: int = 128,
    weight_decay: float = 1e-4,
    seed: int = 42,
    mask_ratio: float = 0.75,
    save_dir: str | None = "checkpoints",
    save_interval: int = 40,
    data_root: str = "data",
    num_workers: int = 4,
    model_kwargs: Optional[dict] = None,
    logger: Any = None,
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    setup_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = (model_kwargs or {}).get("image_size", 32)

    transform = Compose(
        [
            Resize((img_size, img_size)),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model = MAE_ViT(**(model_kwargs or {}))
    print("Model Parameters:", count_parameters(model))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_func  # , verbose=True
    )

    print("Starting to train")
    train_losses, val_losses, step_times = [], [], []

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        step_times = []

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for step, (image, _) in progress_bar:
            start_time = time.time()
            image = image.to(device)

            optimizer.zero_grad()
            out, mask = model(image)
            loss = calculate_mae_loss(
                preds=out, image=image, mask=mask, mask_ratio=mask_ratio
            )

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())
            step_times.append((time.time() - start_time) * 1000)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "step_time": f"{(time.time() - start_time) * 1000:.4F}ms",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                }
            )

            # step-wise logging
            global_step += 1
            if logger is not None:
                logger.log_metrics(
                    {
                        "train/step_loss": float(loss.item()),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            val_progress_bar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation {epoch+1}/{epochs}",
            )
            for _, (val_image, _) in val_progress_bar:
                val_image = val_image.to(device)
                out, mask = model(val_image)
                val_loss = calculate_mae_loss(
                    preds=out, image=val_image, mask=mask, mask_ratio=mask_ratio
                )
                epoch_val_losses.append(val_loss.item())

                if logger is not None:
                    logger.log_metrics(
                        {
                            "val/step_loss": float(val_loss.item()),
                        },
                        step=global_step,
                    )

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append(avg_val_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )
        lr_scheduler.step()

        if save_dir and (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"mae_vit_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved at: {save_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "step_times_ms": step_times,
        "last_lr": lr_scheduler.get_last_lr()[0],
    }
