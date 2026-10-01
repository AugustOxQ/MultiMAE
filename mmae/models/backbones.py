"""Image and text towers.

encode() returns hidden states for the decoders (optionally on masked inputs); embed() returns the
L2-normalized joint-space embedding of a clean input, used by the contrastive loss and retrieval.
The towers call HF CLIP's internal stages directly (embeddings, encoder, norms) so that masking can
happen on the input; tests/test_backbones.py pins them to HF's own outputs.
"""
from __future__ import annotations

from typing import Callable, Iterator, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPConfig, CLIPModel
from transformers.masking_utils import create_causal_mask

# A random tiny CLIP with CLIP's real vocabulary, special tokens and image size, so it works with the
# real tokenizer and preprocessing. Used by tests and CPU smoke runs.
TINY_CLIP = "tiny-random-clip"
POOLINGS = ("native", "mean")


def tiny_clip_config() -> CLIPConfig:
    return CLIPConfig(
        text_config=dict(
            vocab_size=49408,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=77,
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=49407,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=32,
        ),
        projection_dim=24,
    )


def load_clip(pretrained: str) -> CLIPModel:
    if pretrained == TINY_CLIP:
        return CLIPModel(tiny_clip_config())
    return CLIPModel.from_pretrained(pretrained)


def _check_pooling(pooling: str) -> None:
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")


class ClipVisionTower(nn.Module):
    """CLIP ViT with MAE-style token dropping: only CLS and the visible patches enter the transformer."""

    def __init__(self, clip: CLIPModel, pooling: str = "native") -> None:
        super().__init__()
        _check_pooling(pooling)
        self.model = clip.vision_model
        self.projection = clip.visual_projection
        config = self.model.config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.pooling = pooling
        self.mean_projection = (
            nn.Linear(self.hidden_size, clip.projection_dim, bias=False) if pooling == "mean" else None
        )

    def encode(self, pixel_values: torch.Tensor, ids_keep: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 1 + N_keep, hidden) hidden states of CLS and the visible patches (all if ids_keep is None)."""
        hidden = self.model.embeddings(pixel_values)  # (B, 1 + N, C), position embeddings already added
        if ids_keep is not None:
            index = ids_keep.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            hidden = torch.cat([hidden[:, :1], torch.gather(hidden[:, 1:], 1, index)], dim=1)
        hidden = self.model.pre_layrnorm(hidden)
        return self.model.encoder(inputs_embeds=hidden).last_hidden_state

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(pixel_values)
        if self.pooling == "native":
            z = self.projection(self.model.post_layernorm(tokens[:, 0]))
        else:
            z = self.mean_projection(tokens.mean(dim=1))
        return F.normalize(z, dim=-1)

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()
        yield from self.projection.parameters()

    def native_head_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters used only by native pooling (unused without the contrastive loss or with mean pooling)."""
        yield from self.model.post_layernorm.parameters()
        yield from self.projection.parameters()


class ClipTextTower(nn.Module):
    """CLIP text transformer with a learned [MASK] embedding replacing masked tokens on the input."""

    def __init__(self, clip: CLIPModel, pooling: str = "native") -> None:
        super().__init__()
        _check_pooling(pooling)
        self.model = clip.text_model
        self.projection = clip.text_projection
        config = self.model.config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_positions = config.max_position_embeddings
        # CLIP's tokenizer has no [MASK] token, so masked positions get this learned embedding.
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.mask_embedding, std=0.02)
        self.pooling = pooling
        self.mean_projection = (
            nn.Linear(self.hidden_size, clip.projection_dim, bias=False) if pooling == "mean" else None
        )

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, T, hidden) final-layer-normed hidden states; token_mask (B, T) bool marks masked inputs."""
        tokens = self.model.embeddings.token_embedding(input_ids)
        if token_mask is not None:
            tokens = torch.where(token_mask.unsqueeze(-1), self.mask_embedding.to(tokens.dtype), tokens)
        hidden = self.model.embeddings(inputs_embeds=tokens)  # adds position embeddings
        causal = create_causal_mask(
            config=self.model.config, inputs_embeds=hidden, attention_mask=attention_mask, past_key_values=None
        )
        hidden = self.model.encoder(inputs_embeds=hidden, attention_mask=causal, is_causal=True).last_hidden_state
        return self.model.final_layer_norm(hidden)

    def eos_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Index of the pooled (EOS) token, located exactly as HF's CLIPTextModel does."""
        if self.model.eos_token_id == 2:  # legacy configs: EOS has the largest id
            return input_ids.to(torch.int).argmax(dim=-1)
        return (input_ids == self.model.eos_token_id).int().argmax(dim=-1)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(input_ids, attention_mask)
        if self.pooling == "native":
            rows = torch.arange(tokens.shape[0], device=tokens.device)
            z = self.projection(tokens[rows, self.eos_positions(input_ids).to(tokens.device)])
        else:
            weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
            z = self.mean_projection((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0))
        return F.normalize(z, dim=-1)

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()
        yield from self.projection.parameters()

    def native_head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection.parameters()


class Towers(NamedTuple):
    vision: ClipVisionTower
    text: ClipTextTower
    logit_scale: nn.Parameter


def build_hf_clip(pretrained: str, pooling: str) -> Towers:
    clip = load_clip(pretrained)
    return Towers(ClipVisionTower(clip, pooling), ClipTextTower(clip, pooling), clip.logit_scale)


BACKBONES: dict[str, Callable[[str, str], Towers]] = {"hf_clip": build_hf_clip}


def build_backbone(kind: str, pretrained: str, pooling: str = "native") -> Towers:
    if kind not in BACKBONES:
        raise ValueError(f"unknown backbone type {kind!r}; choose from {sorted(BACKBONES)}")
    _check_pooling(pooling)
    return BACKBONES[kind](pretrained, pooling)
