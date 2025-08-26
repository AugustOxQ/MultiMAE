#!/usr/bin/env python3
"""
测试 COCO 数据集的功能
"""

import torch
from torch.utils.data import DataLoader
from src.dataset import COCOImageTextDataset


def test_coco_dataset():
    """测试 COCO 数据集的基本功能"""
    print("=== 测试 COCO 数据集 ===")

    # 创建数据集
    try:
        train_dataset = COCOImageTextDataset(
            root="/data/SSD/coco/images/",
            split="train",
            image_size=224,
            tokenizer_name="bert-base-uncased",
            max_len=64,
        )
        print(f"训练集大小: {len(train_dataset)}")

        val_dataset = COCOImageTextDataset(
            root="/data/SSD/coco/images/",
            split="val",
            image_size=224,
            tokenizer_name="bert-base-uncased",
            max_len=64,
        )
        print(f"验证集大小: {len(val_dataset)}")

    except Exception as e:
        print(f"数据集创建失败: {e}")
        return

    # 测试数据加载
    try:
        # 获取一个样本
        image, token_ids = train_dataset[0]
        print(f"图像形状: {image.shape}")
        print(f"Token IDs 形状: {token_ids.shape}")
        print(f"Token IDs 类型: {token_ids.dtype}")

        # 检查 tokenizer
        tokenizer = train_dataset.tokenizer
        caption = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"解码后的文本: {caption}")

        # 检查 mask 位置
        mask_positions = (token_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        print(f"Mask token 位置: {mask_positions}")

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 测试 DataLoader
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # 避免多进程问题
        )

        print("\n=== 测试 DataLoader ===")
        for batch_idx, (images, token_ids) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  图像形状: {images.shape}")
            print(f"  Token IDs 形状: {token_ids.shape}")

            # 解码第一个样本的文本
            caption = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            print(f"  第一个样本文本: {caption}")

            if batch_idx >= 2:  # 只测试前3个batch
                break

        print("DataLoader 测试成功！")

    except Exception as e:
        print(f"DataLoader 测试失败: {e}")
        return

    print("\n✅ COCO 数据集测试完成！")


if __name__ == "__main__":
    test_coco_dataset()
