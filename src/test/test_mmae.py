#!/usr/bin/env python3
"""
简单的 MMAE 多任务训练测试脚本
"""

import torch
from src.model import MultiModalMAE
from src.metrics.losses import calculate_mae_loss, calculate_mlm_loss
from src.metrics.contrastive import clip_contrastive_loss


def test_mmae_training():
    """测试 MMAE 多任务训练的基本功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

        # 创建模型
    mmae_model = MultiModalMAE(
        image_size=32,
        patch_size=2,
        emb_dim=192,
        encoder_layer=2,
        encoder_head=3,
        mask_ratio=0.75,
        backbone_vision=None,
        text_backbone=None,
        proj_dim=128,
    ).to(device)
    
    print(
        f"MultiModalMAE Model Parameters: {sum(p.numel() for p in mmae_model.parameters()):,}"
    )

    # 创建模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32).to(device)
    token_ids = torch.randint(0, 1000, (batch_size, 16)).to(device)

    # 测试前向传播
    print("\n=== 测试前向传播 ===")

        # MAE 重建
    mae_output, mae_mask = mmae_model.vision(images)
    print(f"MAE output shape: {mae_output.shape}")
    print(f"MAE mask shape: {mae_mask.shape}")
    
    # MLM 重建
    mlm_output, mlm_mask = mmae_model.text(token_ids)
    print(f"MLM output shape: {mlm_output.shape}")
    print(f"MLM mask shape: {mlm_mask.shape}")

    # 对比学习特征
    img_feat = mmae_model.encode_image(images)
    txt_feat = mmae_model.encode_text(token_ids)
    print(f"Image features shape: {img_feat.shape}")
    print(f"Text features shape: {txt_feat.shape}")

    # 测试损失计算
    print("\n=== 测试损失计算 ===")

    mae_loss = calculate_mae_loss(mae_output, images, mae_mask, mask_ratio=0.75)
    mlm_loss = calculate_mlm_loss(mlm_output, token_ids, mlm_mask)
    contrastive_loss, _, _ = clip_contrastive_loss(img_feat, txt_feat, temperature=0.07)

    print(f"MAE Loss: {mae_loss.item():.4f}")
    print(f"MLM Loss: {mlm_loss.item():.4f}")
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")

    # 测试多任务损失
    total_loss = 1.0 * mae_loss + 1.0 * mlm_loss + 1.0 * contrastive_loss
    print(f"Total Loss: {total_loss.item():.4f}")

    # 测试反向传播
    print("\n=== 测试反向传播 ===")

    # 优化器
    optimizer = torch.optim.AdamW(mmae_model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("反向传播成功！")
    print("MMAE 多任务训练测试完成 ✅")


if __name__ == "__main__":
    test_mmae_training()
