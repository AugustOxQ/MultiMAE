# MultiMAE

## 项目简介
MultiMAE 是一个用于多模态自监督学习与检索评测的研究型项目，包含：
- 基于 ViT 的图像 MAE 预训练与重建
- 基于 Transformer/HuggingFace 的文本 MLM 预训练
- 图文对比学习（CLIP 风格）
- 简单的“视觉-语言”融合 MAE（融合后进行图像补丁重建与可选的文本重建）

项目支持使用 Hydra 进行配置管理，并集成 WandB 进行训练与评估指标记录。

## 目标与特性
- 统一的训练与评估入口：`main_mae.py`、`main_mlm.py`、`main_mmae.py`、`main_fusion_mmae.py`
- 数据集适配：内置 COCO 图文数据集的简化读取器与检索评测脚本
- 可替换的骨干与组件：支持 timm ViT、HF 文本 backbone，提供轻量自实现作为回退
- 简洁的度量与损失：MAE/MLM/对比学习损失，支持早停与 wandb 日志

## 代码结构
```text
MultiMAE/
├─ configs/                   # Hydra 配置
│  ├─ mae_config.yaml
│  ├─ mlm_config.yaml
│  ├─ mmae_config.yaml
│  └─ fusion_mmae_config.yaml
├─ src/
│  ├─ dataset/               # 数据集适配
│  │  ├─ coco_dataset.py
│  │  └─ image_dataset.py
│  ├─ eval/                  # 可视化/推理示例
│  │  └─ inference.py
│  ├─ hook/                  # 训练/评估逻辑
│  │  ├─ train_mae.py
│  │  ├─ train_mlm.py
│  │  ├─ train_mmae.py
│  │  ├─ train_fusionmmae.py
│  │  └─ eval_fusionmmae.py  # 检索评测(t2i/i2t)
│  ├─ metrics/               # 损失与对比学习
│  │  ├─ losses.py
│  │  └─ contrastive.py
│  ├─ model/                 # 模型与组件
│  │  ├─ mae.py
│  │  ├─ mlm.py
│  │  ├─ mmae.py
│  │  └─ clip.py             # 基于 CLIP 的封装（可选）
│  └─ utils/                 # 实用工具
│     ├─ seed.py
│     ├─ params.py
│     ├─ data.py
│     └─ wandb_logger.py
├─ main_mae.py               # 图像 MAE 入口
├─ main_mlm.py               # 文本 MLM 入口
├─ main_mmae.py              # 图文多任务 MAE 入口
├─ main_fusion_mmae.py       # 图文融合 MAE 入口
├─ scripts/
│  └─ run_fusion_mmae.sh
└─ requirements.txt
```

## 数据集与准备
- COCO 图文检索与训练使用的默认路径：
  - 图片根目录：`/data/SSD/coco/images/`
  - 标注文件：`/data/SSD/coco/annotations/`
  - 训练：`coco_karpathy_train.json`
  - 验证（单 caption）：`coco_karpathy_val_one_caption.json`
  - 测试（单 caption）：`coco_karpathy_test_one_caption.json`
  - 检索评测脚本会在 val/test 使用 5 captions 的原始 JSON

如需更改路径，请在相应的 `configs/*.yaml` 或数据集构造参数中修改。

## 安装
```bash
pip install -e .
pip install -r requirements.txt
```

建议使用 Conda/venv 隔离环境，确保 PyTorch 与 CUDA 版本对应。

## 快速开始
1) 仅图像 MAE：
```bash
python main_mae.py
```

2) 文本 MLM：
```bash
python main_mlm.py
```

3) 图文多任务 MAE（并行任务，含对比学习）：
```bash
python main_mmae.py
```

4) 图文融合 MAE（先编码再融合，基于融合特征进行重建与检索评测）：
```bash
python main_fusion_mmae.py
```

Hydra 参数可在命令行覆盖，例如：
```bash
python main_fusion_mmae.py train.lr=1e-4 train.batch_size=128 model.backbone=vit_tiny_patch16_224
```

## 评测与日志
- 检索评测：`src/hook/eval_fusionmmae.py` 提供 t2i/i2t Recall、mean/median Rank、mAP 等指标
- 训练日志：若在配置中启用 `wandb.enabled=true`，将自动记录到指定项目

## 许可证
本项目仅用于学术研究与开发演示，具体许可证以仓库根目录为准（如未提供请联系维护者）。
