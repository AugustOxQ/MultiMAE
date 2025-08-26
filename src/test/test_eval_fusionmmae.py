import os
from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.model.mmae import MultiModalFusionMAE
from src.utils import setup_seed, count_parameters
from src.dataset import MSCOCOTestDataset
from src.hook.eval_fusionmmae import evalrank


def train_fusionmmae(
    # ===== 训练基础参数 =====
    batch_size: int = 256,
    seed: int = 42,
    # ===== 数据相关参数 =====
    data_root: str = "/data/SSD/coco/images/",
    image_size: int = 224,
    num_workers: int = 8,
    # ===== 模型架构参数 =====
    # Vision
    backbone_vision: Optional[str] = "vit_tiny_patch16_224",
    # Language
    text_backbone: Optional[str] = "bert-base-uncased",
    text_max_len: int = 64,
    proj_dim: int = 256,
    fusion_method: str = "concat",
):
    setup_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    fusion_model = MultiModalFusionMAE(
        image_size=image_size,
        patch_size=16,
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        mask_ratio=0.75,
        backbone_vision=backbone_vision,
        text_backbone=text_backbone,
        proj_dim=proj_dim,
        fusion_method=fusion_method,
    ).to(device)

    print("MultiModalFusionMAE Model Parameters:", count_parameters(fusion_model))

    print("\n=== 开始额外评估retrieval测试 ===")
    test_set = MSCOCOTestDataset(
        root=data_root,
        split="test",
        image_size=image_size,
        tokenizer_name=str(text_backbone),
        max_len=text_max_len,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    metrics = evalrank(fusion_model, test_loader)
    print(f"Test Retrieval Metrics: {metrics}")


if __name__ == "__main__":
    train_fusionmmae()
