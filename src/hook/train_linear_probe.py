import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR

from src.model import MAE_ViT, ViT_Classifier
from src.utils import setup_seed, count_parameters


def train_linear_probe(
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 12,
    weight_decay: float = 1e-4,
    seed: int = 42,
    pretrained: bool = True,
    path_to_model: str = "",
    save_dir: str = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    setup_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        "data", train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.CIFAR10(
        "data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    mae_model = MAE_ViT()
    if pretrained and path_to_model is not None:
        mae_model.load_state_dict(torch.load(path_to_model))
    linear_probe = ViT_Classifier(model=mae_model, num_classes=10).to(device)

    print("Model Parameters:", count_parameters(linear_probe))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(linear_probe.parameters(), lr=lr, weight_decay=weight_decay)
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1)
    )
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(epochs):
        linear_probe.train()
        epoch_train_losses, epoch_train_accuracies = [], []
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )

        for _, (image, labels) in progress_bar:
            start_time = time.time()
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = linear_probe(image)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            accuracy = accuracy_score(labels_np, preds)

            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{accuracy * 100:.2f}%",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                }
            )

        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        linear_probe.eval()
        epoch_val_losses, epoch_val_accuracies = [], []
        with torch.no_grad():
            val_progress_bar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation {epoch+1}/{epochs}",
            )
            for _, (val_image, val_labels) in val_progress_bar:
                val_image = val_image.to(device)
                val_labels = val_labels.to(device)
                logits = linear_probe(val_image)
                val_loss = criterion(logits, val_labels)
                val_preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_labels_np = val_labels.cpu().numpy()
                val_accuracy = accuracy_score(val_labels_np, val_preds)
                epoch_val_losses.append(val_loss.item())
                epoch_val_accuracies.append(val_accuracy)

        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        avg_val_accuracy = sum(epoch_val_accuracies) / len(epoch_val_accuracies)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.4f}, Train accuracy: {avg_train_accuracy * 100:.2f}%, Val loss: {avg_val_loss:.4f}, Val accuracy: {avg_val_accuracy * 100:.2f}%"
        )
        lr_scheduler.step()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
    }
