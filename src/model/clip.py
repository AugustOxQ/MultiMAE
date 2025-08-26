import torch
import numpy as np
from typing import Optional, Tuple

from einops import rearrange
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer


class CLIPVisionEncoder(torch.nn.Module):
    """
    基于CLIP vision encoder的MAE encoder
    不需要shuffle，直接以原本顺序处理图像
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", mask_ratio: float = 0.75
    ) -> None:
        super().__init__()

        # 加载CLIP vision model
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        self.mask_ratio = mask_ratio

        # 获取模型配置信息
        self.config = self.clip_vision.config
        self.emb_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size

        # 冻结CLIP参数（可选，根据需要调整）
        for param in self.clip_vision.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            img: (B, 3, H, W) 输入图像
        Returns:
            features: (T+1, B, C) 特征向量，包含CLS token
            backward_indexes: 用于兼容MAE接口，这里返回None
        """
        # 使用CLIP vision encoder处理图像
        outputs = self.clip_vision(img)
        features = outputs.last_hidden_state  # (B, T+1, C)

        # 转换为 (T+1, B, C) 格式以兼容MAE接口
        features = rearrange(features, "b t c -> t b c")

        # 返回特征和None（因为不需要shuffle）
        return features, None

    def encode_image_clean(self, images: torch.Tensor) -> torch.Tensor:
        """
        获取完整的、有序的图像特征，用于对比学习
        Args:
            images: (B, 3, H, W) 输入图像
        Returns:
            features: (B, T+1, C) 完整的图像特征
        """
        with torch.no_grad():
            outputs = self.clip_vision(images)
            features = outputs.last_hidden_state  # (B, T+1, C)
            return features


class CLIPTextEncoder(torch.nn.Module):
    """
    基于CLIP text encoder的MLM encoder
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", mask_ratio: float = 0.15
    ) -> None:
        super().__init__()

        # 加载CLIP text model和tokenizer
        self.clip_text = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.mask_ratio = mask_ratio

        # 获取模型配置信息
        self.config = self.clip_text.config
        self.vocab_size = self.config.vocab_size
        self.max_seq_len = self.config.max_position_embeddings
        self.emb_dim = self.config.hidden_size

        # 获取特殊token ID
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        # 冻结CLIP参数（可选，根据需要调整）
        for param in self.clip_text.parameters():
            param.requires_grad = False

        # 定义MLM head为可学习参数，确保注册到模型中
        self.mlm_head = torch.nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            token_ids: (B, T) 输入token IDs
        Returns:
            logits: (B, T, V) 词汇表预测logits
            mask: (B, T, 1) mask位置标识
        """
        device = token_ids.device
        B, T = token_ids.shape

        # 构建attention mask
        attention_mask = (token_ids != self.pad_token_id).long()

        # 构建随机mask位置，避免mask pad tokens
        num_to_mask = max(1, int(T * self.mask_ratio))
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        for b in range(B):
            valid_positions = torch.nonzero(attention_mask[b], as_tuple=False).squeeze(
                1
            )
            if valid_positions.numel() == 0:
                continue
            choice = valid_positions[
                torch.randperm(valid_positions.numel(), device=device)[:num_to_mask]
            ]
            mask[b, choice] = True

        # 创建masked input
        masked_input_ids = token_ids.clone()
        masked_input_ids[mask] = self.mask_token_id

        # 使用CLIP text encoder
        outputs = self.clip_text(
            input_ids=masked_input_ids, attention_mask=attention_mask
        )
        features = outputs.last_hidden_state  # (B, T, C)

        # 通过已注册的MLM head
        logits = self.mlm_head(features)  # (B, T, V)

        # 返回mask作为float (B, T, 1)
        mask_float = mask.unsqueeze(-1).float()

        return logits, mask_float

    def encode_text_clean(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        获取完整的、未mask的文本特征，用于对比学习
        Args:
            token_ids: (B, T) 输入token IDs
        Returns:
            features: (B, T, C) 完整的文本特征
        """
        attention_mask = (token_ids != self.pad_token_id).long()

        with torch.no_grad():
            outputs = self.clip_text(input_ids=token_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state  # (B, T, C)
            return features


class CLIPMAEEncoder(torch.nn.Module):
    """
    专门用于MAE的CLIP vision encoder包装器
    保持与TimmViTEncoder相同的接口
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", mask_ratio: float = 0.75
    ) -> None:
        super().__init__()

        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        self.mask_ratio = mask_ratio

        # 获取配置信息
        self.config = self.clip_vision.config
        self.emb_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size

        # 为了兼容性，设置一些属性
        self.old_grid = (
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )

    def forward(
        self, img: torch.Tensor, return_unmasked_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，保持与TimmViTEncoder相同的接口
        Args:
            img: (B, 3, H, W) 输入图像
            return_unmasked_features: 是否返回未mask的特征
        Returns:
            features: (T+1, B, C) 特征向量
            backward_indexes: 兼容性返回，这里返回None
        """
        if return_unmasked_features:
            return self.encode_image_clean(img), None

        # 使用CLIP vision encoder
        outputs = self.clip_vision(img)
        features = outputs.last_hidden_state  # (B, T+1, C)

        # 转换为 (T+1, B, C) 格式
        features = rearrange(features, "b t c -> t b c")

        return features, None

    def encode_image_clean(self, images: torch.Tensor) -> torch.Tensor:
        """
        获取完整的图像特征
        Args:
            images: (B, 3, H, W) 输入图像
        Returns:
            features: (B, T+1, C) 完整的图像特征
        """
        with torch.no_grad():
            outputs = self.clip_vision(images)
            features = outputs.last_hidden_state  # (B, T+1, C)
            return features


class CLIPMLMEncoder(torch.nn.Module):
    """
    专门用于MLM的CLIP text encoder包装器
    保持与HFMLMBackbone相同的接口
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", mask_ratio: float = 0.15
    ) -> None:
        super().__init__()

        self.clip_text = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.mask_ratio = mask_ratio

        # 获取配置信息
        self.config = self.clip_text.config
        self.vocab_size = self.config.vocab_size
        self.max_seq_len = self.config.max_position_embeddings
        self.emb_dim = self.config.hidden_size

        # 获取特殊token ID
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        # 定义MLM head为可学习参数
        self.mlm_head = torch.nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，保持与HFMLMBackbone相同的接口
        Args:
            token_ids: (B, T) 输入token IDs
        Returns:
            logits: (B, T, V) 词汇表预测logits
            mask: (B, T, 1) mask位置标识
        """
        device = token_ids.device
        B, T = token_ids.shape

        # 构建attention mask
        attention_mask = (token_ids != self.pad_token_id).long()

        # 构建随机mask位置
        num_to_mask = max(1, int(T * self.mask_ratio))
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)

        for b in range(B):
            valid_positions = torch.nonzero(attention_mask[b], as_tuple=False).squeeze(
                1
            )
            if valid_positions.numel() == 0:
                continue
            choice = valid_positions[
                torch.randperm(valid_positions.numel(), device=device)[:num_to_mask]
            ]
            mask[b, choice] = True

        # 创建masked input
        masked_input_ids = token_ids.clone()
        masked_input_ids[mask] = self.mask_token_id

        # 使用CLIP text encoder
        outputs = self.clip_text(
            input_ids=masked_input_ids, attention_mask=attention_mask
        )
        features = outputs.last_hidden_state  # (B, T, C)

        # 通过已注册的MLM head
        logits = self.mlm_head(features)  # (B, T, V)

        # 返回mask作为float (B, T, 1)
        mask_float = mask.unsqueeze(-1).float()

        return logits, mask_float

    def encode_text_clean(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        获取完整的、未mask的文本特征
        Args:
            token_ids: (B, T) 输入token IDs
        Returns:
            features: (B, T, C) 完整的文本特征
        """
        attention_mask = (token_ids != self.pad_token_id).long()

        with torch.no_grad():
            outputs = self.clip_text(input_ids=token_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state  # (B, T, C)
            return features
