"""Fusion modules.

Each takes projected image tokens (B, Li, dim), projected text tokens (B, Lt, dim) and the text padding
(B, Lt, True = padded), any of them None for a single-modality model, and returns the memory each
decoder reads. Add a variant by writing one class with this interface and registering it in FUSIONS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class FusionOutput:
    image_memory: torch.Tensor | None
    image_padding: torch.Tensor | None
    text_memory: torch.Tensor | None
    text_padding: torch.Tensor | None


def concat_modalities(
    image: torch.Tensor | None,
    text: torch.Tensor | None,
    text_padding: torch.Tensor | None,
    type_embed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add modality-type embeddings and concatenate along the sequence; returns (sequence, padding)."""
    parts, paddings = [], []
    if image is not None:
        parts.append(image + type_embed[0])
        paddings.append(torch.zeros(image.shape[:2], dtype=torch.bool, device=image.device))
    if text is not None:
        parts.append(text + type_embed[1])
        if text_padding is None:
            text_padding = torch.zeros(text.shape[:2], dtype=torch.bool, device=text.device)
        paddings.append(text_padding)
    if not parts:
        raise ValueError("fusion needs at least one modality")
    return torch.cat(parts, dim=1), torch.cat(paddings, dim=1)


def _type_embedding(dim: int) -> nn.Parameter:
    embed = nn.Parameter(torch.empty(2, dim))
    nn.init.normal_(embed, std=0.02)
    return embed


class NoFusion(nn.Module):
    """Each decoder reads only its own modality (v0's parallel multimodal MAE)."""

    def forward(self, image, text, text_padding) -> FusionOutput:
        return FusionOutput(image, None, text, text_padding)


def append_memory_token(
    memory: torch.Tensor, padding: torch.Tensor | None, token: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one always-visible token (B, dim) to a decoder memory (B, L, dim); padding (B, L) may be None."""
    if padding is None:
        padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
    memory = torch.cat([memory, token.unsqueeze(1).to(memory.dtype)], dim=1)
    padding = torch.cat([padding, torch.zeros_like(padding[:, :1])], dim=1)
    return memory, padding


class ConcatFusion(nn.Module):
    """Concatenate both modalities, then optionally `depth` transformer layers; both decoders read it."""

    def __init__(self, dim: int, depth: int = 0, heads: int = 8) -> None:
        super().__init__()
        self.type_embed = _type_embedding(dim)
        self.encoder = None
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)

    def forward(self, image, text, text_padding) -> FusionOutput:
        x, padding = concat_modalities(image, text, text_padding, self.type_embed)
        if self.encoder is not None:
            x = self.encoder(x, src_key_padding_mask=padding)
        return FusionOutput(x, padding, x, padding)


class _Learner(nn.Module):
    """Transformer encoder plus an output projection (v0's TransformerLearnerHead)."""

    def __init__(self, dim: int, heads: int, depth: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(x, src_key_padding_mask=padding))


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class MultiLearnerFusion(nn.Module):
    """v0's multi-learner design: image, text and joint learners on the concatenated sequence.

    Image memory = MLP([image learner ; joint learner]); text memory = MLP([text learner ; joint learner]).
    """

    def __init__(self, dim: int, heads: int = 8, learner_depth: int = 2, learner_ff_dim: int = 512) -> None:
        super().__init__()
        self.type_embed = _type_embedding(dim)
        self.image_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.text_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.joint_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.image_mlp = _mlp(2 * dim, learner_ff_dim, dim)
        self.text_mlp = _mlp(2 * dim, learner_ff_dim, dim)

    def forward(self, image, text, text_padding) -> FusionOutput:
        x, padding = concat_modalities(image, text, text_padding, self.type_embed)
        joint = self.joint_learner(x, padding)
        image_memory = self.image_mlp(torch.cat([self.image_learner(x, padding), joint], dim=-1))
        text_memory = self.text_mlp(torch.cat([self.text_learner(x, padding), joint], dim=-1))
        return FusionOutput(image_memory, padding, text_memory, padding)


FUSIONS: dict[str, Callable[[Any], nn.Module]] = {
    "none": lambda cfg: NoFusion(),
    "concat": lambda cfg: ConcatFusion(cfg.dim, depth=cfg.depth, heads=cfg.heads),
    "multilearner": lambda cfg: MultiLearnerFusion(
        cfg.dim, heads=cfg.heads, learner_depth=cfg.learner_depth, learner_ff_dim=cfg.learner_ff_dim
    ),
}


def build_fusion(cfg: Any) -> nn.Module:
    if cfg.type not in FUSIONS:
        raise ValueError(f"unknown fusion type {cfg.type!r}; choose from {sorted(FUSIONS)}")
    return FUSIONS[cfg.type](cfg)
