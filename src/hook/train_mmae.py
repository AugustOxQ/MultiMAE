import os
from typing import Optional, Any
import math
import time
import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer

from src.model import MultiModalMAE
from src.metrics.contrastive import clip_contrastive_loss, calculate_contrastive_loss
from src.metrics.losses import calculate_mae_loss, calculate_mlm_loss
from src.utils import setup_seed, count_parameters
from src.dataset import COCOImageTextDataset


def train_mmae(
    # ===== 训练基础参数 =====
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 42,
    # ===== 数据相关参数 =====
    data_root: str = "data",
    image_size: int = 224,
    num_workers: int = 4,
    # ===== 模型架构参数 =====
    # Vision
    backbone_vision: Optional[str] = "vit_tiny_patch16_224",
    # Language
    text_backbone: Optional[str] = "bert-base-uncased",
    text_max_len: int = 16,
    proj_dim: int = 256,
    # ===== 损失权重参数 =====
    mae_weight: float = 1.0,
    mlm_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    temperature: float = 0.07,
    # ===== 保存和日志参数 =====
    save_dir: Optional[str] = None,
    save_interval: int = 5,
    logger: Any = None,
    # ===== Early Stopping 参数 =====
    patience: int = 10,
    min_delta: float = 1e-4,
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_set = COCOImageTextDataset(
        root=data_root,
        split="train",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    val_set = COCOImageTextDataset(
        root=data_root,
        split="val",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # model
    mmae_model = MultiModalMAE(
        image_size=image_size,
        patch_size=16,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        mask_ratio=0.75,
        backbone_vision=backbone_vision,
        text_backbone=text_backbone,
        proj_dim=proj_dim,
    ).to(device)

    print("MultiModalMAE Model Parameters:", count_parameters(mmae_model))

    # Optimizer
    optimizer = torch.optim.AdamW(
        mmae_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    train_losses, val_losses = [], []
    train_mae_losses, train_mlm_losses, train_contrastive_losses = [], [], []
    val_mae_losses, val_mlm_losses, val_contrastive_losses = [], [], []
    global_step = 0

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        mmae_model.train()
        epoch_losses = []
        epoch_mae_losses = []
        epoch_mlm_losses = []
        epoch_contrastive_losses = []
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for _, (images, token_ids) in pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)

            optimizer.zero_grad()

            # 1. MAE reconstruction loss
            mae_output, mae_mask = mmae_model.vision(images)
            mae_loss = calculate_mae_loss(mae_output, images, mae_mask, mask_ratio=0.75)

            # 2. MLM reconstruction loss
            mlm_output, mlm_mask = mmae_model.text(token_ids)
            mlm_loss = calculate_mlm_loss(mlm_output, token_ids, mlm_mask)

            # 3. Contrastive loss
            img_cls, img_masked, img_remain = mmae_model.encode_image_split(images)
            txt_cls, txt_masked, txt_remain = mmae_model.encode_text_split(token_ids)

            (
                contrastive_loss_cls,
                contrastive_loss_mask_remain,
                contrastive_loss_mask_mask,
            ) = calculate_contrastive_loss(
                img_cls,
                txt_cls,
                img_masked,
                txt_masked,
                img_remain,
                txt_remain,
                temperature=temperature,
            )

            contrastive_loss = (
                contrastive_loss_cls
                + contrastive_loss_mask_remain
                + contrastive_loss_mask_mask
            )
            # Combined loss
            total_loss = +mlm_weight * mlm_loss + contrastive_weight * contrastive_loss

            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_mae_losses.append(mae_loss.item())
            epoch_mlm_losses.append(mlm_loss.item())
            epoch_contrastive_losses.append(contrastive_loss.item())
            pbar.set_postfix(
                {
                    "total_loss": f"{total_loss.item():.4f}",
                    "mae_loss": f"{mae_loss.item():.4f}",
                    "mlm_loss": f"{mlm_loss.item():.4f}",
                    "contrastive_cls_loss": f"{contrastive_loss_cls.item():.4f}",
                    "contrastive_mask_remain_loss": f"{contrastive_loss_mask_remain.item():.4f}",
                    "contrastive_mask_mask_loss": f"{contrastive_loss_mask_mask.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                }
            )

            global_step += 1
            if logger is not None:
                logger.log_metrics(
                    {
                        "train/step_total_loss": float(total_loss.item()),
                        "train/step_mae_loss": float(mae_loss.item()),
                        "train/step_mlm_loss": float(mlm_loss.item()),
                        "train/step_contrastive_cls_loss": float(
                            contrastive_loss_cls.item()
                        ),
                        "train/step_contrastive_mask_remain_loss": float(
                            contrastive_loss_mask_remain.item()
                        ),
                        "train/step_contrastive_mask_mask_loss": float(
                            contrastive_loss_mask_mask.item()
                        ),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

        avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        train_losses.append(avg_train_loss)

        # Calculate epoch averages for individual losses
        avg_mae_loss = sum(epoch_mae_losses) / max(1, len(epoch_mae_losses))
        avg_mlm_loss = sum(epoch_mlm_losses) / max(1, len(epoch_mlm_losses))
        avg_contrastive_loss = sum(epoch_contrastive_losses) / max(
            1, len(epoch_contrastive_losses)
        )

        train_mae_losses.append(avg_mae_loss)
        train_mlm_losses.append(avg_mlm_loss)
        train_contrastive_losses.append(avg_contrastive_loss)

        # validation
        mmae_model.eval()
        val_epoch_losses = []
        val_epoch_mae_losses = []
        val_epoch_mlm_losses = []
        val_epoch_contrastive_losses = []
        with torch.no_grad():
            vpbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation {epoch+1}/{epochs}",
            )
            for _, (images, token_ids) in vpbar:
                images = images.to(device)
                token_ids = token_ids.to(device)

                # Validation losses
                mae_output, mae_mask = mmae_model.vision(images)
                mae_loss = calculate_mae_loss(
                    mae_output, images, mae_mask, mask_ratio=0.75
                )

                mlm_output, mlm_mask = mmae_model.text(token_ids)
                mlm_loss = calculate_mlm_loss(mlm_output, token_ids, mlm_mask)

                img_feat = mmae_model.encode_image(images)
                txt_feat = mmae_model.encode_text(token_ids)
                contrastive_loss, _, _ = clip_contrastive_loss(
                    img_feat, txt_feat, temperature=temperature
                )

                val_total_loss = (
                    mae_weight * mae_loss
                    + mlm_weight * mlm_loss
                    + contrastive_weight * contrastive_loss
                )
                val_epoch_losses.append(val_total_loss.item())
                val_epoch_mae_losses.append(mae_loss.item())
                val_epoch_mlm_losses.append(mlm_loss.item())
                val_epoch_contrastive_losses.append(contrastive_loss.item())

                if logger is not None:
                    logger.log_metrics(
                        {
                            "val/step_total_loss": float(val_total_loss.item()),
                            "val/step_mae_loss": float(mae_loss.item()),
                            "val/step_mlm_loss": float(mlm_loss.item()),
                            "val/step_contrastive_loss": float(contrastive_loss.item()),
                        },
                        step=global_step,
                    )

        avg_val_loss = sum(val_epoch_losses) / max(1, len(val_epoch_losses))
        val_losses.append(avg_val_loss)

        # Calculate validation epoch averages for individual losses
        avg_val_mae_loss = sum(val_epoch_mae_losses) / max(1, len(val_epoch_mae_losses))
        avg_val_mlm_loss = sum(val_epoch_mlm_losses) / max(1, len(val_epoch_mlm_losses))
        avg_val_contrastive_loss = sum(val_epoch_contrastive_losses) / max(
            1, len(val_epoch_contrastive_losses)
        )

        val_mae_losses.append(avg_val_mae_loss)
        val_mlm_losses.append(avg_val_mlm_loss)
        val_contrastive_losses.append(avg_val_contrastive_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = mmae_model.state_dict().copy()
            print(f"Epoch {epoch+1}: 新的最佳验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"Epoch {epoch+1}: 验证损失未改善，patience: {patience_counter}/{patience}"
            )

        # Check if we should stop early
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break

        print(
            f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )
        lr_scheduler.step()

        if save_dir and (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"mmae_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "mmae_model": mmae_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"Model checkpoint saved at: {save_path}")

    # # Restore best model if early stopping was used
    # if best_model_state is not None:
    #     print(f"恢复最佳模型（验证损失: {best_val_loss:.4f}）")
    #     mmae_model.load_state_dict(best_model_state)

    #     # Save best model
    #     if save_dir:
    #         best_model_path = os.path.join(save_dir, "mmae_best_model.pth")
    #         torch.save(
    #             {
    #                 "mmae_model": best_model_state,
    #                 "best_val_loss": best_val_loss,
    #                 "epoch": epoch - patience_counter + 1,
    #             },
    #             best_model_path,
    #         )
    #         print(f"最佳模型已保存到: {best_model_path}")

    # Test set evaluation
    print("\n=== 开始测试集评估 ===")
    test_set = COCOImageTextDataset(
        root=data_root,
        split="test",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    mmae_model.eval()
    test_losses = []
    test_mae_losses = []
    test_mlm_losses = []
    test_contrastive_losses = []

    with torch.no_grad():
        test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        for _, (images, token_ids) in test_pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)

            # Test losses
            mae_output, mae_mask = mmae_model.vision(images)
            mae_loss = calculate_mae_loss(mae_output, images, mae_mask, mask_ratio=0.75)

            mlm_output, mlm_mask = mmae_model.text(token_ids)
            mlm_loss = calculate_mlm_loss(mlm_output, token_ids, mlm_mask)

            img_feat = mmae_model.encode_image(images)
            txt_feat = mmae_model.encode_text(token_ids)
            contrastive_loss, _, _ = clip_contrastive_loss(
                img_feat, txt_feat, temperature=temperature
            )

            test_total_loss = (
                mae_weight * mae_loss
                + mlm_weight * mlm_loss
                + contrastive_weight * contrastive_loss
            )

            test_losses.append(test_total_loss.item())
            test_mae_losses.append(mae_loss.item())
            test_mlm_losses.append(mlm_loss.item())
            test_contrastive_losses.append(contrastive_loss.item())

            test_pbar.set_postfix(
                {
                    "test_total_loss": f"{test_total_loss.item():.4f}",
                    "test_mae_loss": f"{mae_loss.item():.4f}",
                    "test_mlm_loss": f"{mlm_loss.item():.4f}",
                    "test_contrastive_loss": f"{contrastive_loss.item():.4f}",
                }
            )

            if logger is not None:
                logger.log_metrics(
                    {
                        "test/step_total_loss": float(test_total_loss.item()),
                        "test/step_mae_loss": float(mae_loss.item()),
                        "test/step_mlm_loss": float(mlm_loss.item()),
                        "test/step_contrastive_loss": float(contrastive_loss.item()),
                    },
                    step=global_step,
                )

    # Calculate test averages
    avg_test_loss = sum(test_losses) / max(1, len(test_losses))
    avg_test_mae_loss = sum(test_mae_losses) / max(1, len(test_mae_losses))
    avg_test_mlm_loss = sum(test_mlm_losses) / max(1, len(test_mlm_losses))
    avg_test_contrastive_loss = sum(test_contrastive_losses) / max(
        1, len(test_contrastive_losses)
    )

    print(f"\n=== 测试集结果 ===")
    print(f"Test Total Loss: {avg_test_loss:.4f}")
    print(f"Test MAE Loss: {avg_test_mae_loss:.4f}")
    print(f"Test MLM Loss: {avg_test_mlm_loss:.4f}")
    print(f"Test Contrastive Loss: {avg_test_contrastive_loss:.4f}")

    return {
        "train_total_losses": train_losses,
        "val_total_losses": val_losses,
        "train_mae_losses": train_mae_losses,
        "train_mlm_losses": train_mlm_losses,
        "train_contrastive_losses": train_contrastive_losses,
        "val_mae_losses": val_mae_losses,
        "val_mlm_losses": val_mlm_losses,
        "val_contrastive_losses": val_contrastive_losses,
        "test_total_loss": avg_test_loss,
        "test_mae_loss": avg_test_mae_loss,
        "test_mlm_loss": avg_test_mlm_loss,
        "test_contrastive_loss": avg_test_contrastive_loss,
        "last_lr": lr_scheduler.get_last_lr()[0],
        # Early stopping info
        "best_val_loss": best_val_loss,
        "early_stopped": patience_counter >= patience,
        "final_epoch": epoch + 1,
    }
