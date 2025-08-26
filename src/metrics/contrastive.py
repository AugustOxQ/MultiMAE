import torch
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute CLIP-style bidirectional contrastive loss.
    - image_features: (B, D)
    - text_features:  (B, D)
    Returns: (loss, logits_per_image, logits_per_text)
    """
    assert image_features.shape == text_features.shape, "Feature shapes must match"
    image_features = l2_normalize(image_features)
    text_features = l2_normalize(text_features)

    logits_per_image = image_features @ text_features.t() / temperature  # (B, B)
    logits_per_text = logits_per_image.t()

    targets = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    loss = (loss_i + loss_t) * 0.5
    return loss, logits_per_image, logits_per_text


def calculate_contrastive_loss(
    img_cls: torch.Tensor,
    txt_cls: torch.Tensor,
    img_masked: torch.Tensor,
    txt_masked: torch.Tensor,
    img_remain: torch.Tensor,
    txt_remain: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    contrastive_loss_cls, _, _ = clip_contrastive_loss(
        img_cls, txt_cls, temperature=temperature
    )

    img_mask_txt_remain = torch.cat([img_masked, txt_remain], dim=0)
    txt_mask_img_remain = torch.cat([txt_masked, img_remain], dim=0)

    contrastive_loss_mask_remain, _, _ = clip_contrastive_loss(
        img_mask_txt_remain, txt_mask_img_remain, temperature=temperature
    )

    img_mask_txt_mask = torch.cat([img_masked, txt_masked], dim=0)
    txt_mask_img_mask = torch.cat([txt_masked, img_masked], dim=0)

    contrastive_loss_mask_mask, _, _ = clip_contrastive_loss(
        img_mask_txt_mask, txt_mask_img_mask, temperature=temperature
    )

    return (
        contrastive_loss_cls,
        contrastive_loss_mask_remain,
        contrastive_loss_mask_mask,
    )
