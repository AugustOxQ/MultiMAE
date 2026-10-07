"""COCO decoder probes for H-b (spec sections 10 and 11): ECCV Caption i2t queries, per-query AP@R, dual-encoder
top-k and re-ranking, decoder caption scores, and image blends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from eccv_caption import Metrics

from mmae.data.coco import retrieval_items
from mmae.engine.eccv import CAPTIONS_FILE, check_ids, map_coco_ids


@dataclass
class EccvI2T:
    query_image: np.ndarray  # (Q,) row in the 5k test order
    positives: list[np.ndarray]  # caption rows (image-major, i * 5 + j)
    R: np.ndarray  # (Q,) number of listed positives, as the package counts them


def eccv_i2t(annotations_dir: str | Path) -> EccvI2T:
    annotations_dir = Path(annotations_dir)
    items = retrieval_items(annotations_dir, "test")
    image_ids, caption_ids = map_coco_ids(items, annotations_dir / CAPTIONS_FILE)
    metrics = Metrics()
    check_ids(image_ids, caption_ids, metrics)
    image_row = {int(i): n for n, i in enumerate(image_ids)}
    caption_row = {int(c): n for n, c in enumerate(caption_ids.reshape(-1))}
    gts = metrics.eccv_gts["i2t"]
    queries = sorted(gts)
    return EccvI2T(
        query_image=np.array([image_row[int(q)] for q in queries]),
        positives=[np.array(sorted(caption_row[int(p)] for p in set(gts[q]) if int(p) in caption_row)) for q in queries],
        R=np.array([len(set(gts[q])) for q in queries]),
    )


def ap_at_r(order: np.ndarray, positives: list[np.ndarray], R: np.ndarray) -> np.ndarray:
    out = np.zeros(len(R))
    for i, r in enumerate(R):
        rel = np.isin(order[i, :r], positives[i]).astype(float)
        out[i] = (np.cumsum(rel) / np.arange(1, r + 1) * rel).sum() / r
    return out


def topk_captions(image_emb: torch.Tensor, caption_emb: torch.Tensor, query_image: np.ndarray, k: int) -> np.ndarray:
    image = torch.nn.functional.normalize(image_emb.float(), dim=-1)[torch.as_tensor(query_image)]
    captions = torch.nn.functional.normalize(caption_emb.float(), dim=-1)
    out = []
    for start in range(0, len(image), 256):
        scores = image[start : start + 256] @ captions.T
        out.append(torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k])
    return torch.cat(out).numpy()


def rerank(order: np.ndarray, top_scores: np.ndarray) -> np.ndarray:
    k = top_scores.shape[1]
    within = np.argsort(-top_scores, axis=1, kind="stable")
    out = order.copy()
    out[:, :k] = np.take_along_axis(order[:, :k], within, axis=1)
    return out
