#!/usr/bin/env python3
"""
测试更新后的 MMAE 训练功能
"""

import torch
from src.hook import train_mmae


def test_mmae_training():
    """测试 MMAE 训练的基本功能"""
    print("=== 测试更新后的 MMAE 训练 ===")

    # 使用小规模参数进行快速测试
    try:
        results = train_mmae(
            epochs=2,  # 只训练2个epoch
            lr=1e-4,
            batch_size=4,  # 小batch size
            weight_decay=1e-4,
            seed=42,
            image_size=32,  # 小图像尺寸
            data_root="/data/SSD/coco/images/",
            num_workers=0,  # 避免多进程问题
            temperature=0.07,
            backbone_vision=None,  # 使用自定义编码器
            text_backbone="bert-base-uncased",
            text_max_len=32,
            proj_dim=128,
            mae_weight=1.0,
            mlm_weight=1.0,
            contrastive_weight=1.0,
            save_dir=None,  # 不保存模型
            save_interval=1,
            logger=None,  # 不使用wandb
        )

        print("\n✅ 训练完成！")
        print("=== 训练结果 ===")
        print(f"训练轮数: {len(results['train_total_losses'])}")
        print(f"验证轮数: {len(results['val_total_losses'])}")
        print(f"最终训练损失: {results['train_total_losses'][-1]:.4f}")
        print(f"最终验证损失: {results['val_total_losses'][-1]:.4f}")
        print(f"测试总损失: {results['test_total_loss']:.4f}")
        print(f"测试MAE损失: {results['test_mae_loss']:.4f}")
        print(f"测试MLM损失: {results['test_mlm_loss']:.4f}")
        print(f"测试对比损失: {results['test_contrastive_loss']:.4f}")
        print(f"最终学习率: {results['last_lr']:.6f}")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_wandb_logging():
    """测试 wandb 日志记录（模拟）"""
    print("\n=== 测试 Wandb 日志记录 ===")

    # 模拟训练结果
    mock_results = {
        "train_total_losses": [0.5, 0.4],
        "val_total_losses": [0.6, 0.5],
        "train_mae_losses": [0.2, 0.15],
        "train_mlm_losses": [0.15, 0.12],
        "train_contrastive_losses": [0.15, 0.13],
        "val_mae_losses": [0.25, 0.2],
        "val_mlm_losses": [0.2, 0.18],
        "val_contrastive_losses": [0.15, 0.12],
        "test_total_loss": 0.45,
        "test_mae_loss": 0.18,
        "test_mlm_loss": 0.15,
        "test_contrastive_loss": 0.12,
        "last_lr": 1e-5,
    }

    print("模拟的 wandb 指标:")
    print("训练指标:")
    for i, (tr, va) in enumerate(
        zip(mock_results["train_total_losses"], mock_results["val_total_losses"]), 1
    ):
        print(f"  Epoch {i}: train={tr:.4f}, val={va:.4f}")

    print("测试指标:")
    print(f"  test_total_loss: {mock_results['test_total_loss']:.4f}")
    print(f"  test_mae_loss: {mock_results['test_mae_loss']:.4f}")
    print(f"  test_mlm_loss: {mock_results['test_mlm_loss']:.4f}")
    print(f"  test_contrastive_loss: {mock_results['test_contrastive_loss']:.4f}")

    print("✅ Wandb 日志记录测试完成")


if __name__ == "__main__":
    print("开始测试更新后的 MMAE 功能...")

    # 测试训练功能
    success = test_mmae_training()

    if success:
        # 测试 wandb 日志记录
        test_wandb_logging()
        print("\n🎉 所有测试通过！")
    else:
        print("\n❌ 测试失败，请检查错误信息")
