"""MultiMAE: CLIP towers with input masking, a fusion module and query decoders.

forward(batch) returns every loss, so the training loop and DDP see one ordinary forward pass:
  1. clean pass (both modalities only): contrastive loss on the towers' joint-space embeddings;
  2. masked pass: mask patches and tokens, encode the visible inputs, fuse, decode, and score
     MAE on masked patches and MLM on masked tokens.
With reconstruction false (the contrastive baseline) only step 1 runs, and the projections, fusion and
decoders are not built.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from mmae.losses import contrastive_loss, mae_loss, mlm_loss
from mmae.models.backbones import build_backbone
from mmae.models.decoders import QueryDecoder
from mmae.models.fusion import append_memory_token, build_fusion
from mmae.models.masking import random_patch_mask, random_token_mask

# Parameter names (substrings) that get no weight decay, besides every parameter with ndim < 2.
NO_DECAY_KEYS = ("position_embedding", "token_embedding", "queries", "pos_embed", "type_embed")
MODALITIES = {"image", "text"}
MLM_IMAGE_SOURCES = ("masked", "clean", "clean_detached")


class MultiMAE(nn.Module):
    def __init__(self, cfg: DictConfig, max_text_len: int) -> None:
        super().__init__()
        modalities = set(cfg.modalities)
        if not modalities or modalities - MODALITIES:
            raise ValueError(f"modalities must be a non-empty subset of {sorted(MODALITIES)}, got {cfg.modalities}")
        self.use_image = "image" in modalities
        self.use_text = "text" in modalities
        if cfg.fusion.type == "multilearner" and len(modalities) < 2:
            raise ValueError("multilearner fusion needs both modalities")
        self.has_contrastive = self.use_image and self.use_text
        # .get: run configs saved before the flag existed have no key and mean the full model
        self.reconstruction = bool(cfg.get("reconstruction", True))
        if not self.reconstruction and not self.has_contrastive:
            raise ValueError("reconstruction=false (contrastive only) needs both modalities")
        self.image_ratio = float(cfg.masking.image_ratio)
        self.text_ratio = float(cfg.masking.text_ratio)
        self.loss_weights = {k: float(v) for k, v in cfg.loss.weights.items()}
        self.gather = bool(cfg.loss.gather)
        self.norm_pix = bool(cfg.loss.norm_pix)
        # .get: run configs saved before a switch existed mean its default (today's behaviour)
        self.mlm_image_source = str(cfg.get("mlm_image_source", "masked"))
        if self.mlm_image_source not in MLM_IMAGE_SOURCES:
            raise ValueError(f"mlm_image_source must be one of {MLM_IMAGE_SOURCES}, got {self.mlm_image_source!r}")
        if self.mlm_image_source != "masked" and not (
            self.reconstruction and self.has_contrastive and cfg.fusion.type != "none"
        ):
            raise ValueError(
                "mlm_image_source=clean* needs both modalities, reconstruction and a fusion through which the "
                "text decoder reads the image (concat or multilearner)"
            )
        self.text_mode = str(cfg.masking.get("text_mode", "random"))
        if self.text_mode not in ("random", "content"):
            raise ValueError(f"masking.text_mode must be 'random' or 'content', got {self.text_mode!r}")

        towers = build_backbone(cfg.backbone.type, cfg.backbone.pretrained, cfg.pooling)
        dim, dec = int(cfg.fusion.dim), cfg.decoder
        self.vision = None
        self.text = None
        self.logit_scale = towers.logit_scale if self.has_contrastive else None
        if self.use_image:
            self.vision = towers.vision
            if self.reconstruction:
                self.image_proj = nn.Linear(self.vision.hidden_size, dim)
                self.image_decoder = QueryDecoder(
                    self.vision.num_patches, dim, self.vision.patch_size**2 * 3,
                    depth=dec.depth, heads=dec.heads, dropout=dec.dropout,
                )
        if self.use_text:
            if max_text_len > towers.text.max_positions:
                raise ValueError(f"max_text_len {max_text_len} exceeds the text tower's {towers.text.max_positions}")
            self.text = towers.text
            if self.reconstruction:
                self.text_proj = nn.Linear(self.text.hidden_size, dim)
                self.text_decoder = QueryDecoder(
                    max_text_len, dim, self.text.vocab_size, depth=dec.depth, heads=dec.heads, dropout=dec.dropout
                )
        if self.reconstruction:
            self.fusion = build_fusion(cfg.fusion)
        self.pooled_conditioning = bool(cfg.get("pooled_conditioning", False))
        if self.pooled_conditioning:
            if not (self.reconstruction and self.has_contrastive):
                raise ValueError("pooled_conditioning needs both modalities and reconstruction")
            embed_dim = towers.vision.projection.out_features
            self.pooled_image_proj = nn.Linear(embed_dim, dim)  # clean image embedding -> text-decoder memory token
            self.pooled_text_proj = nn.Linear(embed_dim, dim)   # clean text embedding -> image-decoder memory token

        for tower in self.towers():
            # The native contrastive head is unused without the contrastive loss or under mean pooling,
            # and the mean projection is unused without the contrastive loss; freeze them so DDP does
            # not wait for gradients that never arrive.
            if not self.has_contrastive or cfg.pooling == "mean":
                for param in tower.native_head_parameters():
                    param.requires_grad = False
            if not self.has_contrastive and tower.mean_projection is not None:
                for param in tower.mean_projection.parameters():
                    param.requires_grad = False
            if not self.reconstruction and hasattr(tower, "mask_embedding"):
                tower.mask_embedding.requires_grad = False  # only the masked pass uses it
            if cfg.freeze_backbones:
                for param in tower.pretrained_parameters():
                    param.requires_grad = False

    def towers(self) -> list[nn.Module]:
        return [t for t in (self.vision, self.text) if t is not None]

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision.embed(pixel_values)

    def embed_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text.embed(input_ids, attention_mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch.get("pixel_values")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        losses: dict[str, torch.Tensor] = {}

        image_emb = text_emb = clean_image_tokens = None
        if self.has_contrastive:
            clean_image_tokens = self.vision.encode(images)
            image_emb = self.vision.pool(clean_image_tokens)
            text_emb = self.embed_text(input_ids, attention_mask)
            losses["contrastive"] = contrastive_loss(image_emb, text_emb, self.logit_scale, self.gather)

        if not self.reconstruction:
            return {"loss": self.loss_weights["contrastive"] * losses["contrastive"],
                    "loss_contrastive": losses["contrastive"]}

        image_tokens = text_tokens = text_padding = masked_text_hidden = None
        if self.use_image:
            ids_keep, patch_mask = random_patch_mask(
                images.shape[0], self.vision.num_patches, self.image_ratio, device=images.device
            )
            image_tokens = self.image_proj(self.vision.encode(images, ids_keep))
        if self.use_text:
            allowed = None
            if self.text_mode == "content":
                if "content_tokens_mask" not in batch:
                    raise KeyError(
                        "masking.text_mode=content needs batch['content_tokens_mask']: build the Collator with "
                        "content_words=True"
                    )
                allowed = batch["content_tokens_mask"]
            token_mask = random_token_mask(attention_mask, batch["special_tokens_mask"], self.text_ratio, allowed=allowed)
            masked_text_hidden = self.text.encode(input_ids, attention_mask, token_mask)
            text_tokens = self.text_proj(masked_text_hidden)
            text_padding = ~attention_mask.bool()

        fused = self.fusion(image_tokens, text_tokens, text_padding)
        text_fused = fused
        if self.mlm_image_source != "masked":  # M1: the MLM decoder reads every patch of the clean pass
            clean = clean_image_tokens.detach() if self.mlm_image_source == "clean_detached" else clean_image_tokens
            text_fused = self.fusion(self.image_proj(clean), text_tokens, text_padding)
        image_memory, image_padding = fused.image_memory, fused.image_padding
        text_memory, text_memory_padding = text_fused.text_memory, text_fused.text_padding
        if self.pooled_conditioning:  # M3: each decoder also reads the OTHER modality's clean pooled embedding
            if self.use_image:
                image_memory, image_padding = append_memory_token(image_memory, image_padding, self.pooled_text_proj(text_emb))
            if self.use_text:
                text_memory, text_memory_padding = append_memory_token(
                    text_memory, text_memory_padding, self.pooled_image_proj(image_emb)
                )
        if self.use_image:
            pred = self.image_decoder(image_memory, image_padding)
            losses["mae"] = mae_loss(pred, images, patch_mask, self.vision.patch_size, norm_pix=self.norm_pix)
        if self.use_text:
            logits = self.text_decoder(text_memory, text_memory_padding, query_padding=text_padding)
            losses["mlm"] = mlm_loss(logits, input_ids, token_mask)

        out = {f"loss_{name}": value for name, value in losses.items()}
        out["loss"] = sum(self.loss_weights[name] * value for name, value in losses.items())
        return out

    def param_groups(self, lr: float, lr_backbone: float, weight_decay: float) -> list[dict]:
        """AdamW groups: pretrained tower weights at lr_backbone, everything else at lr; no decay on
        biases, norms, embeddings, queries and the logit scale."""
        backbone = {id(p) for tower in self.towers() for p in tower.pretrained_parameters()}
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim < 2 or any(key in name for key in NO_DECAY_KEYS)
            groups.setdefault((id(param) in backbone, no_decay), []).append(param)
        return [
            {
                "params": params,
                "lr": lr_backbone if is_backbone else lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{'backbone' if is_backbone else 'head'}_{'no_decay' if no_decay else 'decay'}",
            }
            for (is_backbone, no_decay), params in sorted(groups.items())
        ]
