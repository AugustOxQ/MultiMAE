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

from src.model.mmae import MultiModalFusionMAE
from src.model.mmae import MultiModalFusionMAE_CLIP
from src.metrics.contrastive import clip_contrastive_loss
from src.metrics.losses import calculate_mae_loss, calculate_mlm_loss
from src.utils import setup_seed, count_parameters
from src.dataset import COCOImageTextDataset, MSCOCOTestDataset
from src.hook.eval_fusionmmae import evalrank
from accelerate import Accelerator


def train_fusionmmae(
    # ===== 训练基础参数 =====
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 256,
    eval_batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 42,
    # ===== 数据相关参数 =====
    data_root: str = "data",
    image_size: int = 224,
    num_workers: int = 8,
    # ===== 模型架构参数 =====
    # Vision
    backbone_vision: str = "openai/clip-vit-base-patch32",
    # Language
    text_backbone: str = "openai/clip-vit-base-patch32",
    text_max_len: int = 16,
    proj_dim: int = 256,
    fusion_method: str = "concat",
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
    accelerator: Optional[Accelerator] = None,
):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    setup_seed(seed)
    accelerator = accelerator or Accelerator()
    device = accelerator.device

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
        val_set, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )

    retrieval_val_set = MSCOCOTestDataset(
        root=data_root,
        split="val",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    retrieval_val_loader = DataLoader(
        retrieval_val_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # model
    fusion_model = MultiModalFusionMAE_CLIP(
        image_size=image_size,
        patch_size=16,
        emb_dim=768,
        decoder_layer=4,
        decoder_head=8,
        mask_ratio=0.75,
        backbone_vision=backbone_vision,
        text_backbone=text_backbone,
        proj_dim=proj_dim,
        fusion_method=fusion_method,
    ).to(device)

    accelerator.print(
        "MultiModalFusionMAE Model Parameters:", count_parameters(fusion_model)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / epochs * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # Prepare for distributed/mixed-precision
    fusion_model, optimizer, train_loader, val_loader, retrieval_val_loader = (
        accelerator.prepare(
            fusion_model, optimizer, train_loader, val_loader, retrieval_val_loader
        )
    )

    train_losses, val_losses = [], []
    train_mae_losses, train_mlm_losses, train_contrastive_losses = [], [], []
    val_mae_losses, val_mlm_losses, val_contrastive_losses = [], [], []
    global_step = 0

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        fusion_model.train()
        epoch_losses = []
        epoch_mae_losses = []
        epoch_mlm_losses = []
        epoch_contrastive_losses = []
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{epochs}",
            disable=not accelerator.is_local_main_process,
        )
        for _, (images, token_ids) in pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)

            optimizer.zero_grad()

            # Forward pass through MultiModalFusionMAE
            outputs = fusion_model(images, token_ids)

            # 1. MAE reconstruction loss (from image patches)
            # 使用正确的重建损失计算
            fused_tokens = outputs["fused_tokens"]
            mae_loss = fusion_model.calculate_mae_reconstruction_loss(
                images, fused_tokens, mask_ratio=0.75
            )

            # 2. MLM reconstruction loss (if language decoder is available)
            mlm_loss = torch.tensor(0.0, device=device)
            if "text_outputs" in outputs:
                text_logits, text_mask = outputs["text_outputs"]
                mlm_loss = calculate_mlm_loss(text_logits, token_ids, text_mask)

            # 3. Simplified contrastive loss (only CLS tokens)
            img_cls = fusion_model.encode_image_tokens_cls(images)
            txt_cls = fusion_model.encode_text_tokens_cls(token_ids)
            contrastive_loss, _, _ = clip_contrastive_loss(
                img_cls, txt_cls, temperature=temperature
            )

            # Combined loss
            total_loss = (
                mae_weight * mae_loss
                + mlm_weight * mlm_loss
                + contrastive_weight * contrastive_loss
            )

            accelerator.backward(total_loss)
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
                    "contrastive_loss": f"{contrastive_loss.item():.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}",
                }
            )

            global_step += 1
            if accelerator.is_main_process and logger is not None:
                logger.log_metrics(
                    {
                        "train/step_total_loss": float(total_loss.item()),
                        "train/step_mae_loss": float(mae_loss.item()),
                        "train/step_mlm_loss": float(mlm_loss.item()),
                        "train/step_contrastive_loss": float(contrastive_loss.item()),
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
        fusion_model.eval()
        val_epoch_losses = []
        val_epoch_mae_losses = []
        val_epoch_mlm_losses = []
        val_epoch_contrastive_losses = []
        with torch.no_grad():
            vpbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Validation {epoch+1}/{epochs}",
                disable=not accelerator.is_local_main_process,
            )
            for _, (images, token_ids) in vpbar:
                images = images.to(device)
                token_ids = token_ids.to(device)

                # Validation forward pass
                outputs = fusion_model(images, token_ids)

                # Validation losses
                fused_tokens = outputs["fused_tokens"]
                mae_loss = fusion_model.calculate_mae_reconstruction_loss(
                    images, fused_tokens, mask_ratio=0.75
                )

                mlm_loss = torch.tensor(0.0, device=device)
                if "text_outputs" in outputs:
                    text_logits, text_mask = outputs["text_outputs"]
                    mlm_loss = calculate_mlm_loss(text_logits, token_ids, text_mask)

                img_feat = fusion_model.encode_image_tokens_cls(images)
                txt_feat = fusion_model.encode_text_tokens_cls(token_ids)
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

                if accelerator.is_main_process and logger is not None:
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

        accelerator.print("\n=== 开始额外评估val set retrieval测试 ===")

        retrieval_metrics = evalrank(
            fusion_model, retrieval_val_loader, accelerator=accelerator
        )
        accelerator.print(f"Retrieval Val Metrics: {retrieval_metrics}")

        if accelerator.is_main_process and logger is not None:
            for key, value in retrieval_metrics.items():
                logger.log_metrics(
                    {
                        f"val_retrieval/{key}": value,
                    },
                    step=global_step,
                )

        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = fusion_model.state_dict().copy()
            accelerator.print(f"Epoch {epoch+1}: 新的最佳验证损失: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            accelerator.print(
                f"Epoch {epoch+1}: 验证损失未改善，patience: {patience_counter}/{patience}"
            )

        # Check if we should stop early
        if patience_counter >= patience:
            accelerator.print(f"Early stopping triggered after {epoch+1} epochs!")
            break

        accelerator.print(
            f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}"
        )
        lr_scheduler.step()

        if (
            accelerator.is_main_process
            and save_dir
            and (epoch + 1) % save_interval == 0
        ):
            save_path = os.path.join(save_dir, f"fusion_mmae_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "fusion_model": fusion_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path,
            )
            accelerator.print(f"Model checkpoint saved at: {save_path}")

    # Test set evaluation
    accelerator.print("\n=== 开始测试集评估 ===")
    test_set = COCOImageTextDataset(
        root=data_root,
        split="test",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )

    # prepare test loader
    test_loader = accelerator.prepare(test_loader)

    fusion_model.eval()
    test_losses = []
    test_mae_losses = []
    test_mlm_losses = []
    test_contrastive_losses = []

    with torch.no_grad():
        test_pbar = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="Testing",
            disable=not accelerator.is_local_main_process,
        )
        for _, (images, token_ids) in test_pbar:
            images = images.to(device)
            token_ids = token_ids.to(device)

            # Test forward pass
            outputs = fusion_model(images, token_ids)

            # Test losses
            fused_tokens = outputs["fused_tokens"]
            mae_loss = fusion_model.calculate_mae_reconstruction_loss(
                images, fused_tokens, mask_ratio=0.75
            )

            mlm_loss = torch.tensor(0.0, device=device)
            if "text_outputs" in outputs:
                text_logits, text_mask = outputs["text_outputs"]
                mlm_loss = calculate_mlm_loss(text_logits, token_ids, text_mask)

            img_feat = fusion_model.encode_image_tokens_cls(images)
            txt_feat = fusion_model.encode_text_tokens_cls(token_ids)
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

            if accelerator.is_main_process and logger is not None:
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

    accelerator.print(f"\n=== 测试集结果 ===")
    accelerator.print(f"Test Total Loss: {avg_test_loss:.4f}")
    accelerator.print(f"Test MAE Loss: {avg_test_mae_loss:.4f}")
    accelerator.print(f"Test MLM Loss: {avg_test_mlm_loss:.4f}")
    accelerator.print(f"Test Contrastive Loss: {avg_test_contrastive_loss:.4f}")

    accelerator.print("\n=== 开始额外评估retrieval测试 ===")
    retrieval_test_set = MSCOCOTestDataset(
        root=data_root,
        split="test",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    retrieval_test_loader = DataLoader(
        retrieval_test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    retrieval_metrics = evalrank(
        fusion_model, retrieval_test_loader, accelerator=accelerator
    )
    accelerator.print(f"Retrieval Test Metrics: {retrieval_metrics}")

    if accelerator.is_main_process and logger is not None:
        for key, value in retrieval_metrics.items():
            logger.log_metrics(
                {
                    f"test_retrieval/{key}": value,
                },
                step=global_step,
            )

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
