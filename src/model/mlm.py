import torch
import numpy as np

from einops import rearrange, repeat
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
from typing import Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM


class MLM_Encoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
        emb_dim: int = 256,
        num_layer: int = 6,
        num_head: int = 4,
        mask_ratio: float = 0.15,
    ) -> None:
        super().__init__()

        self.token_embed = torch.nn.Embedding(vocab_size, emb_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(max_seq_len, 1, emb_dim))
        self.mask_ratio = mask_ratio

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: (B, T)
        B, T = token_ids.shape
        x = self.token_embed(token_ids)  # (B, T, C)
        x = rearrange(x, "b t c -> t b c")  # (T, B, C)
        x = x + self.pos_embedding[:T]

        # Standard MLM: just add CLS token and process
        x = torch.cat([self.cls_token.expand(-1, x.shape[1], -1), x], dim=0)
        x = rearrange(x, "t b c -> b t c")
        features = self.layer_norm(self.transformer(x))
        features = rearrange(features, "t b c -> t b c")

        # Return features and a simple mask (all positions except CLS)
        mask = torch.ones((T, B, 1), device=features.device, dtype=features.dtype)
        return features, mask


class MLM_Decoder(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
        emb_dim: int = 256,
        num_layer: int = 4,
        num_head: int = 4,
    ) -> None:
        super().__init__()

        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(max_seq_len + 1, 1, emb_dim)
        )

        self.transformer = torch.nn.Sequential(
            *[Block(emb_dim, num_head) for _ in range(num_layer)]
        )
        self.head = torch.nn.Linear(emb_dim, vocab_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        # features: (T, B, C) where T includes cls
        # mask: (T, B, 1) indicating which positions to predict

        # Add positional embeddings
        features = features + self.pos_embedding[: features.shape[0]]

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # drop cls

        logits = self.head(features)  # (T, B, V)

        # Get logits for vocabulary prediction

        # reshape to (B, T, ...)
        logits = rearrange(logits, "t b v -> b t v")
        mask = rearrange(mask, "t b c -> b t c")

        return logits, mask


class MLM_Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
        emb_dim: int = 256,
        encoder_layer: int = 6,
        encoder_head: int = 4,
        decoder_layer: int = 4,
        decoder_head: int = 4,
        mask_ratio: float = 0.15,
        backbone: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.hf_backbone = None
        if backbone:
            if AutoTokenizer is None or AutoModelForMaskedLM is None:
                raise ImportError(
                    "transformers is required to use a pre-trained language backbone. Please install transformers."
                )
            self.hf_backbone = HFMLMBackbone(model_name=backbone, mask_ratio=mask_ratio)
        else:
            self.encoder = MLM_Encoder(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                emb_dim=emb_dim,
                num_layer=encoder_layer,
                num_head=encoder_head,
                mask_ratio=mask_ratio,
            )

            self.decoder = MLM_Decoder(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                emb_dim=emb_dim,
                num_layer=decoder_layer,
                num_head=decoder_head,
            )

    def forward(self, token_ids: torch.Tensor):
        if self.hf_backbone is not None:
            logits, mask = self.hf_backbone(token_ids)
            return logits, mask
        else:
            features, mask = self.encoder(token_ids)
            logits, mask = self.decoder(features, mask)
            return logits, mask


class HFMLMBackbone(torch.nn.Module):
    """
    Wrapper around a HuggingFace Masked LM. Applies token masking and decodes with the pre-trained head.
    This bypasses the custom encoder/decoder and returns (logits, mask) directly.
    """

    def __init__(
        self, model_name: str = "bert-base-uncased", mask_ratio: float = 0.15
    ) -> None:
        super().__init__()
        if AutoTokenizer is None or AutoModelForMaskedLM is None:
            raise ImportError("transformers is required to use HFMLMBackbone")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.mask_ratio = mask_ratio

        # try to get mask and pad token ids
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError(f"Tokenizer {model_name} does not have a mask_token_id")
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

    def forward(self, token_ids: torch.Tensor):
        # token_ids: (B, T) of already tokenized ids
        device = token_ids.device
        B, T = token_ids.shape

        # build attention mask (1 for real tokens)
        attention_mask = (token_ids != self.pad_token_id).long()

        # build random mask positions, avoid masking pad tokens
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

        masked_input_ids = token_ids.clone()
        masked_input_ids[mask] = self.mask_token_id

        outputs = self.model(input_ids=masked_input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # return mask as float (B, T, 1)
        mask_float = mask.unsqueeze(-1).float()
        return logits, mask_float

    def encode_text_clean(self, token_ids: torch.Tensor) -> torch.Tensor:
        """获取完整的、未mask的文本特征，用于对比学习"""
        # 直接使用原始token_ids，不进行masking
        attention_mask = (token_ids != self.pad_token_id).long()

        # 关键：使用原始的、未mask的输入
        outputs = self.model.bert(
            input_ids=token_ids,  # 原始token_ids，不是masked的
            attention_mask=attention_mask,
        )
        feature = outputs.last_hidden_state  # (B, T+1, C)

        return feature
