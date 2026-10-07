"""Training losses: contrastive InfoNCE (optionally over the global batch), MAE and MLM."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

# Upper bound on exp(logit_scale). The trainer clamps the parameter to [0, log(MAX_LOGIT_SCALE)] after each
# optimizer step, as open_clip does. Clamping exp() inside the loss instead would lose the gradient: CLIP's
# pretrained logit_scale (4.605170249938965) exponentiates to 100.0000076 on CPU, so it is zero from the first
# step; on CUDA it is exactly 100.0, so it is lost once the parameter drifts above ln 100 and never returns.
MAX_LOGIT_SCALE = 100.0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def contrastive_loss(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor, gather: bool = True
) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalized embeddings, matched pairs on the diagonal.

    With `gather` and more than one process, each rank scores its local rows against the global batch
    (open_clip's local-loss formulation); gradients flow back through the gather. Every rank must have
    the same local batch size. The scale is exp(logit_scale), unclamped: the trainer bounds the parameter
    itself after each optimizer step (MAX_LOGIT_SCALE).
    """
    scale = logit_scale.float().exp()
    image_emb, text_emb = image_emb.float(), text_emb.float()
    batch = image_emb.shape[0]
    if gather and _world_size() > 1:
        all_image = torch.cat(all_gather(image_emb), dim=0)
        all_text = torch.cat(all_gather(text_emb), dim=0)
        offset = dist.get_rank() * batch
    else:
        all_image, all_text, offset = image_emb, text_emb, 0
    labels = torch.arange(batch, device=image_emb.device) + offset
    with torch.autocast(device_type=image_emb.device.type, enabled=False):  # autocast would recast the matmul to bf16
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


def mlm_emotion_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, token_mask: torch.Tensor,
    emotion_logits: torch.Tensor, emotion: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MLM with the annotator's emotion as one more hidden target per caption (H-b spec 2026-10-07, 5.1): the
    mean cross-entropy over the masked tokens and the emotion slots together. Also returns, for logging only,
    loss_mlm_tokens (mean over masked tokens, 0 if none) and loss_emotion (mean over captions)."""
    if token_mask.any():
        token_ce = F.cross_entropy(logits[token_mask].float(), input_ids[token_mask], reduction="sum")
    else:
        token_ce = logits.float().sum() * 0.0
    emotion_ce = F.cross_entropy(emotion_logits.float(), emotion, reduction="sum")
    n_tokens, n_captions = token_mask.sum(), emotion.shape[0]
    total = (token_ce + emotion_ce) / (n_tokens + n_captions)
    parts = {
        "loss_mlm_tokens": (token_ce / n_tokens.clamp(min=1)).detach(),
        "loss_emotion": (emotion_ce / n_captions).detach(),
    }
    return total, parts
