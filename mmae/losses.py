"""Training losses: contrastive InfoNCE (optionally over the global batch), MAE and MLM."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

MAX_LOGIT_SCALE = 100.0  # CLIP clamps exp(logit_scale) at 100


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def contrastive_loss(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor, gather: bool = True
) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalized embeddings, matched pairs on the diagonal.

    With `gather` and more than one process, each rank scores its local rows against the global batch
    (open_clip's local-loss formulation); gradients flow back through the gather. Every rank must have
    the same local batch size.
    """
    scale = logit_scale.float().exp().clamp(max=MAX_LOGIT_SCALE)
    image_emb, text_emb = image_emb.float(), text_emb.float()
    batch = image_emb.shape[0]
    if gather and _world_size() > 1:
        all_image = torch.cat(all_gather(image_emb), dim=0)
        all_text = torch.cat(all_gather(text_emb), dim=0)
        offset = dist.get_rank() * batch
    else:
        all_image, all_text, offset = image_emb, text_emb, 0
    labels = torch.arange(batch, device=image_emb.device) + offset
    logits_image = scale * image_emb @ all_text.T
    logits_text = scale * text_emb @ all_image.T
    return 0.5 * (F.cross_entropy(logits_image, labels) + F.cross_entropy(logits_text, labels))


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, p*p*C), patches in the ViT's row-major order, pixels as (p, p, C)."""
    b, c, h, w = images.shape
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch_size * patch_size * c)


def mae_loss(
    pred: torch.Tensor, images: torch.Tensor, mask: torch.Tensor, patch_size: int, norm_pix: bool = True
) -> torch.Tensor:
    """Mean over masked patches of the per-patch MSE between predicted and target pixels."""
    target = patchify(images.float(), patch_size)
    if norm_pix:  # normalize each target patch by its own mean and variance, as in the MAE paper
        target = (target - target.mean(dim=-1, keepdim=True)) / (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
    per_patch = ((pred.float() - target) ** 2).mean(dim=-1)
    weights = mask.to(per_patch.dtype)
    return (per_patch * weights).sum() / weights.sum().clamp(min=1.0)


def mlm_loss(logits: torch.Tensor, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at masked positions only; zero (with a gradient path) if nothing is masked."""
    if not token_mask.any():
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits[token_mask].float(), input_ids[token_mask])
