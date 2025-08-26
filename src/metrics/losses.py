import torch


def calculate_mae_loss(
    preds: torch.Tensor, image: torch.Tensor, mask: torch.Tensor, mask_ratio: float
) -> torch.Tensor:
    return torch.mean((preds - image) ** 2 * mask) / mask_ratio


def calculate_mlm_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Cross-entropy loss computed only on masked positions.
    logits: (B, T, V)
    labels: (B, T) token ids
    mask:   (B, T, 1) or (B, T) where True/1 indicates masked positions
    """
    if mask.dim() == 3:
        mask_bool = mask.squeeze(-1).bool()
    else:
        mask_bool = mask.bool()

    ignore_index = -100
    effective_labels = labels.clone()
    effective_labels[~mask_bool] = ignore_index

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        effective_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss
