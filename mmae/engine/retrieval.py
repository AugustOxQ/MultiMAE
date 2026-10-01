"""COCO retrieval evaluation (image-to-text and text-to-image) on joint-space embeddings."""
from __future__ import annotations

import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderStateMixin

RECALL_KS = (1, 5, 10)


@torch.no_grad()
def encode_retrieval_set(
    model: torch.nn.Module, loader, accelerator: Accelerator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image embeddings (N, D) and caption embeddings (N, K, D) in dataset order.

    With more than one process the loader must be prepared (sharded); embeddings are gathered batch
    by batch with captions kept as (B, K, D), so accelerate drops the samples it duplicated to even
    out the last batch.
    """
    if accelerator is not None and accelerator.num_processes > 1 and not isinstance(loader, DataLoaderStateMixin):
        raise ValueError(
            "encode_retrieval_set needs a loader prepared by accelerate when running on "
            f"{accelerator.num_processes} processes; pass accelerator.prepare(loader)"
        )
    net = accelerator.unwrap_model(model) if accelerator is not None else model
    was_training = net.training
    net.eval()
    images, captions = [], []
    for batch in loader:
        b, k, t = batch["input_ids"].shape
        image = net.embed_image(batch["pixel_values"])
        caption = net.embed_text(
            batch["input_ids"].reshape(b * k, t), batch["attention_mask"].reshape(b * k, t)
        ).view(b, k, -1)
        if accelerator is not None:
            image = accelerator.gather_for_metrics(image)
            caption = accelerator.gather_for_metrics(caption)
        images.append(image.float())
        captions.append(caption.float())
    net.train(was_training)
    return torch.cat(images), torch.cat(captions)


def _ranks(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """1-based rank of each positive: 1 + number of candidates scored strictly higher.

    scores (Q, M), positives (Q, P) candidate indices -> (Q, P).
    """
    positive_scores = scores.gather(1, positives)
    return 1 + (scores.unsqueeze(1) > positive_scores.unsqueeze(-1)).sum(dim=-1)


def retrieval_metrics(image_emb: torch.Tensor, caption_emb: torch.Tensor, chunk: int = 256) -> dict[str, float]:
    """Recall@{1,5,10}, mean/median rank, mAP for i2t and t2i, and rsum (sum of the six recalls).

    i2t: an image query is correct at k if any of its K captions ranks <= k; its mean/median rank and
    mAP use all K captions. t2i: each caption has one positive image. Recalls and mAP are in percent.
    """
    n, k, d = caption_emb.shape
    device = image_emb.device
    text_emb = caption_emb.reshape(n * k, d)
    caption_ids = torch.arange(n * k, device=device).view(n, k)
    i2t, t2i = [], []
    for start in range(0, n, chunk):
        scores = image_emb[start : start + chunk] @ text_emb.T
        i2t.append(_ranks(scores, caption_ids[start : start + chunk]))
    for start in range(0, n * k, chunk):
        stop = min(start + chunk, n * k)
        scores = text_emb[start:stop] @ image_emb.T
        image_ids = (torch.arange(start, stop, device=device) // k).unsqueeze(1)
        t2i.append(_ranks(scores, image_ids).squeeze(1))
    i2t_ranks, t2i_ranks = torch.cat(i2t).double(), torch.cat(t2i).double()

    best = i2t_ranks.min(dim=1).values
    sorted_ranks = i2t_ranks.sort(dim=1).values
    # clamp the j-th best rank to >= j: matches v0's distinct sorted positions when positives tie
    sorted_ranks = torch.maximum(sorted_ranks, torch.arange(1, k + 1, device=device, dtype=torch.double))
    average_precision = (torch.arange(1, k + 1, device=device, dtype=torch.double) / sorted_ranks).mean(dim=1)
    metrics: dict[str, float] = {}
    for top in RECALL_KS:
        metrics[f"i2t_R{top}"] = 100.0 * (best <= top).double().mean().item()
        metrics[f"t2i_R{top}"] = 100.0 * (t2i_ranks <= top).double().mean().item()
    metrics.update(
        i2t_meanR=sorted_ranks.mean().item(),
        i2t_medR=torch.quantile(sorted_ranks.flatten(), 0.5).item(),
        i2t_mAP=100.0 * average_precision.mean().item(),
        t2i_meanR=t2i_ranks.mean().item(),
        t2i_medR=torch.quantile(t2i_ranks, 0.5).item(),
        t2i_mAP=100.0 * (1.0 / t2i_ranks).mean().item(),
    )
    metrics["rsum"] = sum(metrics[f"{direction}_R{top}"] for direction in ("i2t", "t2i") for top in RECALL_KS)
    return metrics


def evaluate_retrieval(model: torch.nn.Module, loader, accelerator: Accelerator | None = None) -> dict[str, float]:
    images, captions = encode_retrieval_set(model, loader, accelerator)
    return retrieval_metrics(images, captions)
