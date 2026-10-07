"""Random masks: MAE-style patch dropping for images and MLM-style token masking for text."""
from __future__ import annotations

import torch


def _uniform(shape: tuple[int, ...], device, generator: torch.Generator | None) -> torch.Tensor:
    """Uniform noise drawn on the generator's device (if any), then moved to `device`."""
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator, device=generator.device).to(device)


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask `int(num_patches * ratio)` random patches per sample (argsort of noise, as in MAE).

    Returns:
        ids_keep: (B, N_keep) long, indices of the visible patches in ascending order.
        mask: (B, N) bool, True where the patch is masked.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"patch mask ratio must be in [0, 1), got {ratio}")
    num_keep = num_patches - int(num_patches * ratio)
    noise = _uniform((batch_size, num_patches), device, generator)
    ids_keep = noise.argsort(dim=1)[:, :num_keep].sort(dim=1).values
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=noise.device)
    mask.scatter_(1, ids_keep, False)
    return ids_keep, mask


def random_token_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask `max(1, round(ratio * n))` tokens of each caption, n = its real, non-special tokens.

    With `allowed` (B, T) bool, the masked tokens are drawn from the allowed ones only (the count is still set by
    n, capped by the number allowed). Captions with no maskable token get no mask. Returns (B, T) bool.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"token mask ratio must be in (0, 1], got {ratio}")
    real = attention_mask.bool() & ~special_tokens_mask.bool()
    maskable = real if allowed is None else real & allowed.bool()
    n = real.sum(dim=1)
    available = maskable.sum(dim=1)
    k = torch.clamp(torch.round(n.float() * ratio), min=1).long()
    k = torch.where(available > 0, torch.minimum(k, available), torch.zeros_like(k))
    noise = _uniform(tuple(maskable.shape), maskable.device, generator)
    noise = noise.masked_fill(~maskable, 2.0)  # non-maskable positions sort after every maskable one
    ranks = noise.argsort(dim=1).argsort(dim=1)
    return ranks < k.unsqueeze(1)
