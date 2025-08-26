import torch
from typing import Optional

from src.model.mae import MAE_ViT, MAE_Encoder
from src.model.mlm import HFMLMBackbone, MLM_Encoder
from einops import rearrange


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiModalMAE(torch.nn.Module):
    """
    Simple multimodal encoder pair (vision + language) with projection heads for contrastive learning.
    Vision uses MAE_ViT (encoder path), Language uses HFMLMBackbone (or a lightweight MLM_Encoder as fallback).
    """

    def __init__(
        self,
        # vision params
        image_size: int = 224,
        patch_size: int = 16,
        emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        mask_ratio: float = 0.75,
        backbone_vision: Optional[str] = None,
        # language params
        text_backbone: Optional[str] = "bert-base-uncased",
        # projection
        proj_dim: int = 256,
    ) -> None:
        super().__init__()

        # vision model (we keep the MAE_ViT but will only use encoder features)
        self.vision = MAE_ViT(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            encoder_layer=encoder_layer,
            encoder_head=encoder_head,
            decoder_layer=4,
            decoder_head=encoder_head,
            mask_ratio=mask_ratio,
            backbone=backbone_vision,
        )

        # language model backbone (HF masked LM)
        if text_backbone:
            self.text = HFMLMBackbone(model_name=text_backbone, mask_ratio=0.15)
            text_out_dim = self.text.model.config.hidden_size
        else:
            # fallback to lightweight encoder; use its emb_dim as output dim
            self.text = MLM_Encoder(emb_dim=emb_dim)
            text_out_dim = emb_dim

        # projection heads bring both modalities to the same dimension
        vision_out_dim = emb_dim
        self.vision_proj = ProjectionHead(
            vision_out_dim, proj_dim
        )  # TODO: 这一层是吃不到reconstruction loss的，后续修改可以改为在vision和language的encoder后直接添加projection head
        self.text_proj = ProjectionHead(text_out_dim, proj_dim)

    def _encode_image_tokens_clean(self, images: torch.Tensor) -> torch.Tensor:
        """统一获取 (B, N+1, C) 图像tokens（兼容不同encoder）。"""
        encoder = self.vision.encoder
        func = getattr(encoder, "encode_image_clean", None)
        if func is not None and callable(func):
            tokens: torch.Tensor = func(images)  # type: ignore[no-redef]
            return tokens
        features, _ = encoder(images)
        tokens = rearrange(features, "t b c -> b t c")
        tokens = torch.as_tensor(tokens)
        return tokens

    def encode_image_split(self, images: torch.Tensor):

        # TODO: 这里以及text部分其实还可以优化，因为image和languageencoder是用的各自现成的代码，而且是专为unimodal reconsturction设计的，所以encoder从一开始就是include了patch的部分，所以这里和contrastive feature不兼容，后续可以优化一下，减少运行时间
        # Obtain [CLS] token representation from encoder output
        features_all = self._encode_image_tokens_clean(images)  # (B, N+1, C)
        features_all = self.vision_proj(features_all)  # (B, N+1, C_v)
        cls = features_all[:, 0, :]  # (B, C_v)

        features = features_all[:, 1:, :]  # (B, N, C_v)

        seq_len = features.shape[1]
        num_masked = int(seq_len * self.vision.mask_ratio)
        indices = torch.randperm(seq_len)

        mask_indices = indices[:num_masked]
        remain_indices = indices[num_masked:]

        features_masked = features[:, mask_indices, :]
        features_remain = features[:, remain_indices, :]

        # mean pooling
        features_masked = features_masked.mean(dim=1)
        features_remain = features_remain.mean(dim=1)

        return cls, features_masked, features_remain

    def encode_text_split(self, token_ids: torch.Tensor):
        if isinstance(self.text, HFMLMBackbone):
            features_all = self.text.encode_text_clean(token_ids)  # (B, T+1, C)
            features_all = self.text_proj(features_all)  # (B, T+1, C_t)
            cls = features_all[:, 0, :]  # (B, C_t)

            features = features_all[:, 1:, :]  # (B, T, C_t)

            seq_len = features.shape[1]
            # num_masked = int(seq_len * self.text.mask_ratio)
            # BUG: 这里其实不够妥当，因为image的所有patch都是有真实对应图像部分的，但是text却是有可能是padding，所以mask_ratio应该要根据text的实际长度来计算
            real_seq_len = (token_ids != self.text.pad_token_id).sum(dim=1)
            real_seq_len = real_seq_len.clamp(min=1)
            num_masked = int(
                real_seq_len.float().mean().item() * self.text.mask_ratio
            )  # 修复：转换为float类型再取mean
            indices = torch.randperm(seq_len)

            mask_indices = indices[:num_masked]
            remain_indices = indices[num_masked:]

            features_masked = features[:, mask_indices, :]
            features_remain = features[:, remain_indices, :]

            # mean pooling
            features_masked = features_masked.mean(dim=1)
            features_remain = features_remain.mean(dim=1)

            return cls, features_masked, features_remain

        else:
            raise NotImplementedError("Only HFMLMBackbone is supported for now")

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_image_tokens_clean(images)
        cls = self.vision_proj(tokens)[:, 0, :]
        return cls

    @torch.no_grad()
    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 兼容 HF 与自定义 encoder
        if isinstance(self.text, HFMLMBackbone):
            features = self.text.encode_text_clean(token_ids)
        else:
            features, _ = self.text(token_ids)
            features = rearrange(features, "t b c -> b t c")
        cls = self.text_proj(features)[:, 0, :]
        return cls


class ImagePatchDecoder(torch.nn.Module):
    """
    简易图像补丁解码器：将融合后的序列特征映射为每个图像补丁的重建输出。
    - 输入: fused_tokens (B, L, C)
    - 输出: (B, N_patches, patch_out_dim)

    目前采用 mean-pool 得到全局向量，再投影到所有补丁；后续可替换为 Transformer 解码器等更强方案。
    """

    def __init__(
        self,
        fused_dim: int,
        num_patches: int,
        patch_out_dim: int,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_out_dim = patch_out_dim
        self.to_patch_template = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, fused_dim),
            torch.nn.GELU(),
            torch.nn.Linear(fused_dim, patch_out_dim),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        # fused_tokens: (B, L, C)
        global_feat = fused_tokens.mean(dim=1)  # (B, C)
        patch_pred = self.to_patch_template(global_feat)  # (B, D)
        patch_pred = patch_pred.unsqueeze(1).expand(
            -1, self.num_patches, -1
        )  # (B, N, D)
        return patch_pred


class MultiModalFusionMAE(torch.nn.Module):
    """
    Multimodal encoder pair (vision + language) with projection heads for contrastive learning.
    不同于 MultiModalMAE，本模型在两模态编码后进行融合（先用 concat），再用融合特征进行重建。

    当前实现：
    - 仅创建 MAE 与 MLM 的 encoder，并用投影头对齐维度
    - 融合方法通过接口保留，默认 concat，后续可替换为 cross-attn/transformer 等
    - 提供一个简易图像补丁解码器用于重建
    - 语言侧解码器留作接口，暂未实现
    """

    def __init__(
        self,
        # vision params
        image_size: int = 224,
        patch_size: int = 16,
        emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        mask_ratio: float = 0.75,
        backbone_vision: Optional[str] = None,
        # language params
        text_backbone: Optional[str] = "bert-base-uncased",
        # projection & fusion
        proj_dim: int = 256,
        fusion_method: str = "concat",
        # image reconstruction params
        image_patch_out_dim: Optional[int] = None,  # 默认 p*p*3
    ) -> None:
        super().__init__()

        # 仅使用 encoder 路径
        self.vision_encoder = MAE_Encoder(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            num_layer=encoder_layer,
            num_head=encoder_head,
            mask_ratio=mask_ratio,
        )

        if text_backbone:
            self.text_encoder = HFMLMBackbone(model_name=text_backbone, mask_ratio=0.15)
            text_out_dim = self.text_encoder.model.config.hidden_size
        else:
            self.text_encoder = MLM_Encoder(emb_dim=emb_dim)
            text_out_dim = emb_dim

        # 投影到同一维度，便于融合
        vision_out_dim = emb_dim
        self.vision_proj = ProjectionHead(vision_out_dim, proj_dim)
        self.text_proj = ProjectionHead(text_out_dim, proj_dim)

        # 融合方式
        self.fusion_method = fusion_method

        # 图像补丁解码器
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        if image_patch_out_dim is None:
            image_patch_out_dim = patch_size * patch_size * 3
        self.image_decoder = ImagePatchDecoder(
            fused_dim=proj_dim,
            num_patches=num_patches,
            patch_out_dim=image_patch_out_dim,
        )

        # 语言侧解码器占位
        self.language_decoder_head: Optional[torch.nn.Module] = None
        # 若使用HF backbone，可获取词表大小与pad id，直接构建方案1解码器
        try:
            if isinstance(self.text_encoder, HFMLMBackbone):
                vocab_size = self.text_encoder.model.config.vocab_size
                pad_id = self.text_encoder.pad_token_id
                self.language_decoder_head = TextMLMDecoderHead(
                    fused_dim=proj_dim,
                    vocab_size=vocab_size,
                    max_seq_len=256,
                    pad_token_id=pad_id,
                )
        except Exception:
            # 若失败则保持为 None，由外部注入
            pass

    # ---------- Encoding ----------
    def encode_image_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """返回包含 [CLS] 的图像 token: (B, N+1, C_proj)"""
        # 这里的 vision_encoder 是 MAE_Encoder，无 encode_image_clean，直接走 forward
        features, _ = self.vision_encoder(images)  # (T+1, B, C)
        img_tokens = rearrange(features, "t b c -> b t c")  # (B, T+1, C)
        img_tokens = self.vision_proj(img_tokens)  # (B, N+1, C_proj)
        return img_tokens

    def encode_image_tokens_cls(self, images: torch.Tensor) -> torch.Tensor:
        """返回图像 CLS token: (B, C_proj)"""
        tokens = self.encode_image_tokens(images)  # (B, N+1, C_proj)
        return tokens[:, 0, :]  # (B, C_proj)

    def encode_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回包含 [CLS] 的文本 token: (B, T+1, C_proj)"""
        if isinstance(self.text_encoder, HFMLMBackbone):
            txt_tokens = self.text_encoder.encode_text_clean(token_ids)  # (B, T, C)
        else:
            features, _ = self.text_encoder(token_ids)  # (T+1, B, C)
            txt_tokens = rearrange(features, "t b c -> b t c")  # (B, T+1, C)
        txt_tokens = self.text_proj(txt_tokens)  # (B, T+1, C_proj)
        return txt_tokens

    def encode_text_tokens_cls(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回文本 CLS token: (B, C_proj)"""
        tokens = self.encode_text_tokens(token_ids)  # (B, T+1, C_proj)
        return tokens[:, 0, :]  # (B, C_proj)

    # ---------- Fusion ----------
    def fuse_features(
        self,
        img_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """
        融合两模态特征：当前默认 concat 沿序列维拼接；
        后续可扩展为 cross-attention、门控相加、FiLM、Transformer 融合等。
        """
        if method is None:
            method = self.fusion_method
        if method == "concat":
            return torch.cat([img_tokens, txt_tokens], dim=1)
        raise NotImplementedError(f"Fusion method '{method}' is not implemented")

    # ---------- Decoding ----------
    def reconstruct_image(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """从融合序列重建图像补丁输出: (B, N_patches, D_out)"""
        return self.image_decoder(fused_tokens)

    def get_original_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        获取原始图像的补丁表示，用于计算重建损失
        返回: (B, N_patches, patch_dim) 其中 patch_dim = patch_size * patch_size * 3
        """
        B, C, H, W = images.shape
        patch_size = 16  # 假设与模型初始化时的 patch_size 一致

        # 将图像分割成 patches
        patches = images.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(
            B, -1, C * patch_size * patch_size
        )  # (B, N_patches, patch_dim)

        return patches

    def calculate_mae_reconstruction_loss(
        self, images: torch.Tensor, fused_tokens: torch.Tensor, mask_ratio: float = 0.75
    ) -> torch.Tensor:
        """
        计算正确的 MAE 重建损失
        - 获取原始图像补丁
        - 生成 mask 模式（与 encoder 一致）
        - 计算被 mask 位置的 MSE 损失
        """
        B, N_patches, patch_dim = (
            fused_tokens.shape[0],
            self.image_decoder.num_patches,
            self.image_decoder.patch_out_dim,
        )

        # 获取原始图像补丁
        original_patches = self.get_original_image_patches(
            images
        )  # (B, N_patches, patch_dim)

        # 生成 mask（与 encoder 的 mask 策略一致）
        num_masked = int(N_patches * mask_ratio)
        mask_indices = torch.randperm(N_patches, device=fused_tokens.device)[
            :num_masked
        ]

        # 重建的补丁
        reconstructed_patches = self.reconstruct_image(
            fused_tokens
        )  # (B, N_patches, patch_dim)

        # 只计算被 mask 位置的损失
        masked_reconstructed = reconstructed_patches[
            :, mask_indices, :
        ]  # (B, num_masked, patch_dim)
        masked_original = original_patches[
            :, mask_indices, :
        ]  # (B, num_masked, patch_dim)

        mae_loss = torch.nn.functional.mse_loss(masked_reconstructed, masked_original)
        return mae_loss

    def reconstruct_text(
        self, fused_tokens: torch.Tensor, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        语言侧重建接口：默认使用MLM解码头（方案1）。
        返回 (logits, mask)
        """
        if self.language_decoder_head is None:
            raise NotImplementedError(
                "Language reconstruction head is not implemented yet"
            )
        return self.language_decoder_head(fused_tokens, token_ids)

    # ---------- Forward ----------
    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> dict:
        """
        返回：
        - fused_tokens: 融合后的序列 (B, L_img+L_txt, C)
        - image_patches: 图像补丁重建输出 (B, N_patches, D_out)
        - text_outputs: 若接入语言解码器，则返回其输出
        """
        img_tokens = self.encode_image_tokens(images)
        txt_tokens = self.encode_text_tokens(token_ids)
        fused_tokens = self.fuse_features(img_tokens, txt_tokens)

        out: dict = {
            "fused_tokens": fused_tokens,
            "image_patches": self.reconstruct_image(fused_tokens),
        }
        if self.language_decoder_head is not None:
            out["text_outputs"] = self.reconstruct_text(fused_tokens, token_ids)
        return out


class TextMLMDecoderHead(torch.nn.Module):
    """
    方案1：MLM风格文本解码头
    - 输入: fused_tokens (B, L, C)
    - 输入: token_ids (B, T) 仅用于对齐长度与构造mask
    - 输出: logits (B, T, V), mask (B, T, 1)

    简化实现：
    - 使用融合序列的mean-pool得到全局向量，与可学习的位置编码相加后，
      通过两层MLP产生每个位置的词表logits。
    - 后续可替换为更强的cross-attention/transformer结构。
    """

    def __init__(
        self,
        fused_dim: int,
        vocab_size: int,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

        self.pos_embed = torch.nn.Parameter(torch.zeros(max_seq_len, fused_dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, fused_tokens: torch.Tensor, token_ids: torch.Tensor):
        # fused_tokens: (B, L, C)
        # token_ids: (B, T)
        B, T = token_ids.shape
        C = fused_tokens.shape[-1]
        # 全局特征
        global_feat = fused_tokens.mean(dim=1)  # (B, C)
        # 取前T个位置编码
        pos = self.pos_embed[:T]  # (T, C)
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, T, C)
        x = global_feat.unsqueeze(1).expand(-1, T, -1) + pos  # (B, T, C)
        logits = self.mlp(x)  # (B, T, V)
        mask = (token_ids != self.pad_token_id).float().unsqueeze(-1)  # (B, T, 1)
        return logits, mask


class TextARDecoderHead(torch.nn.Module):
    """
    方案2：轻量自回归文本解码器（未接入）
    - 以融合特征为条件，做teacher forcing的自回归预测。
    - 简化实现：使用token嵌入 + 融合全局向量作为条件，叠加若干线性层；
      实际可替换为标准TransformerDecoder + causal mask。
    """

    def __init__(
        self,
        vocab_size: int,
        fused_dim: int,
        emb_dim: int = 256,
        hidden_dim: int = 512,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.token_embed = torch.nn.Embedding(vocab_size, emb_dim)
        self.cond = torch.nn.Linear(fused_dim, emb_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, fused_tokens: torch.Tensor, token_ids: torch.Tensor):
        # fused_tokens: (B, L, C)  -> 条件向量使用mean-pool
        # token_ids: (B, T)  -> teacher forcing输入
        B, T = token_ids.shape
        cond_vec = self.cond(fused_tokens.mean(dim=1))  # (B, E)
        tok = self.token_embed(token_ids)  # (B, T, E)
        x = tok + cond_vec.unsqueeze(1)  # (B, T, E)
        logits = self.mlp(x)  # (B, T, V)
        mask = (token_ids != self.pad_token_id).float().unsqueeze(-1)  # (B, T, 1)
        return logits, mask
