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
from mmae.models.fusion import NoFusion
from mmae.models.masking import random_patch_mask, random_token_mask


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


# ---- D1: decoder caption scores and PMI re-ranking (spec section 10) ----
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
VIEW_RATIO = 0.75  # 25% of the patches stay visible (13 of 49)
THIRDS = {"R<=15": (0, 15), "R16-20": (16, 20), "R>=21": (21, 10**9)}  # cuts of stratified_eccv.py


def null_images(n: int) -> torch.Tensor:
    """The null image: all zeros after CLIP normalisation (the dataset mean colour)."""
    return torch.zeros(n, 3, 224, 224)


def _real_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """Real, non-special positions: CLIP captions are BOS ... EOS then padding."""
    n = attention_mask.sum(1)
    real = attention_mask.bool().clone()
    real[:, 0] = False
    real[torch.arange(len(n), device=real.device), n - 1] = False
    return real


@torch.no_grad()
def caption_scores(model, images: torch.Tensor, captions_tok: dict, kind: str, n_samples: int, seed: int,
                   views: torch.Tensor | None = None, token_masks: torch.Tensor | None = None) -> torch.Tensor:
    """Per pair, the mean over samples of the mean log p of the hidden tokens, (B,).

    parallel: every real token hidden; masked-source models average n_samples 25% views (seeds seed + s), clean-source
    and fusion_none models use the full image and one sample. ratio: n_samples random token masks at the model's own
    text ratio (seeds seed + s), each with its own 25% view for masked-source models, the full image otherwise.

    views (K, B, 13) and token_masks (K, B, T) are optional precomputed draws keyed by the caller (by image and by
    caption id), so a pair's score does not depend on how rows are chunked; sample s uses views[s] / token_masks[s].
    Without them the draws come from per-chunk generators seeded seed + s (row-position keyed)."""
    if kind not in ("parallel", "ratio"):
        raise ValueError(f"kind must be parallel or ratio, got {kind!r}")
    device = images.device
    ids = captions_tok["input_ids"].to(device)
    attention = captions_tok["attention_mask"].to(device)
    real = _real_tokens(attention)
    special = attention.bool() & ~real
    viewed = model.mlm_image_source == "masked" and not isinstance(model.fusion, NoFusion)
    n = n_samples if (kind == "ratio" or viewed) else 1
    total = torch.zeros(len(ids), device=device)
    for s in range(n):
        ids_keep = None
        if viewed and views is not None:
            ids_keep = views[s].to(device)
        elif viewed:
            ids_keep, _ = random_patch_mask(len(ids), model.vision.num_patches, VIEW_RATIO, device=device,
                                            generator=torch.Generator().manual_seed(seed + s))
        if kind == "parallel":
            hidden = real
        elif token_masks is not None:
            hidden = token_masks[s].to(device)
        else:
            hidden = random_token_mask(attention, special, model.text_ratio,
                                       generator=torch.Generator().manual_seed(seed + s)).to(device)
        logits, _ = model.decode_text(images, ids, attention, hidden, ids_keep=ids_keep)
        logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        total += (logp * hidden).sum(1) / hidden.sum(1).clamp(min=1)
    return total / n


def _z(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=1, keepdims=True)
    return (x - x.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1.0)


def _order_by(order: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.take_along_axis(order, np.argsort(-scores, axis=1, kind="stable"), axis=1)


def score_a(dec: np.ndarray, null: np.ndarray, alpha: float) -> np.ndarray:
    return dec - alpha * null


def score_b(dual: np.ndarray, dec: np.ndarray, null: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return _z(dual) + beta * _z(dec - alpha * null)


def tune(dual, dec, null, order, positives, R) -> dict:
    """alpha (re-ranker a), and alpha and beta (re-ranker b) maximising mean AP@R over (Q, k) candidates in dual order.
    Ties keep the first grid point, so no correction (alpha 0, beta 0) wins a tie."""
    dual, dec, null, order = (np.asarray(x) for x in (dual, dec, null, order))
    R = np.full(len(order), R) if np.ndim(R) == 0 else np.asarray(R)
    best_a, best_b = (-1.0, 0.0), (-1.0, 0.0, 0.0)
    for alpha in ALPHAS:
        ap = ap_at_r(_order_by(order, score_a(dec, null, alpha)), positives, R).mean()
        if ap > best_a[0]:
            best_a = (ap, alpha)
        for beta in BETAS:
            ap = ap_at_r(_order_by(order, score_b(dual, dec, null, alpha, beta)), positives, R).mean()
            if ap > best_b[0]:
                best_b = (ap, alpha, beta)
    return {"alpha_a": best_a[1], "alpha_b": best_b[1], "beta": best_b[2],
            "val_map_a": float(best_a[0]), "val_map_b": float(best_b[0])}


def r_precision(order: np.ndarray, positives: list[np.ndarray], R: np.ndarray) -> np.ndarray:
    return np.array([np.isin(order[i, :r], positives[i]).sum() / r for i, r in enumerate(R)])


def r_thirds(R: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.where((R >= lo) & (R <= hi))[0] for name, (lo, hi) in THIRDS.items()}


def d1_metrics(order: np.ndarray, q: EccvI2T) -> dict:
    """ECCV mAP@R and R-Precision (percent) of a full per-query caption order, overall and by thirds of R."""
    ap = 100 * ap_at_r(order, q.positives, q.R)
    rp = 100 * r_precision(order, q.positives, q.R)
    by = {name: {"n": int(len(ix)), "map_at_r": float(ap[ix].mean()) if len(ix) else None,
                 "r_precision": float(rp[ix].mean()) if len(ix) else None}
          for name, ix in r_thirds(q.R).items()}
    return {"n": int(len(ap)), "map_at_r": float(ap.mean()), "r_precision": float(rp.mean()), "by_third": by}


def token_mask_table(tok: dict, ratio: float, n_samples: int, seed: int) -> torch.Tensor:
    """(K, N, T) hidden-token sets for every caption of a split, drawn once per sample over the whole table and
    indexed by caption id, so a caption has the same hidden set for every query, the null image and every model
    with the same ratio. tok needs input_ids, attention_mask and special_tokens_mask."""
    return torch.stack([
        random_token_mask(tok["attention_mask"], tok["special_tokens_mask"], ratio,
                          generator=torch.Generator().manual_seed(seed + s))
        for s in range(n_samples)])


# ---- D2: blend probe (spec section 11) ----
LAMBDAS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIX_M = (3, 5, 7, 9, 11)
PATCH_SIZE, N_KEEP = 32, 13
TIE_TOL = 1e-6


def select_pairs(caption_emb: torch.Tensor, n_pairs: int = 1000, n_random: int = 100000, percentile: float = 10,
                 seed_pairs: int = 0, seed_random: int = 1) -> np.ndarray:
    """(n_pairs, 2) distinct image pairs whose mean cross-caption cosine is below the percentile of n_random random
    pairs. caption_emb (N, 5, D); the mean over the 5 x 5 cosines of unit vectors is the dot of the mean unit vectors.
    Candidates are drawn from RandomState(seed_pairs) in order; unordered duplicates are skipped."""
    mean = torch.nn.functional.normalize(caption_emb.float(), dim=-1).mean(1).numpy()
    n = len(mean)

    def draw(rng, count):
        a = rng.randint(0, n, count)
        b = rng.randint(0, n - 1, count)
        return a, b + (b >= a)  # distinct

    a, b = draw(np.random.RandomState(seed_random), n_random)
    threshold = np.percentile((mean[a] * mean[b]).sum(1), percentile)
    rng, seen, out = np.random.RandomState(seed_pairs), set(), []
    for _ in range(1000):
        a, b = draw(rng, max(10 * n_pairs, 1000))
        for x, y, s in zip(a, b, (mean[a] * mean[b]).sum(1)):
            key = (min(x, y), max(x, y))
            if s < threshold and key not in seen:
                seen.add(key)
                out.append((x, y))
                if len(out) == n_pairs:
                    return np.array(out)
    raise ValueError(f"only {len(out)} of {n_pairs} pairs below the {percentile}th percentile were found")


def blend(a: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
    """lam a + (1 - lam) b on the normalised tensors the model sees (lam = 1 is a exactly)."""
    return lam * a + (1 - lam) * b


def patch_mix(a: torch.Tensor, b: torch.Tensor, m: int, generator: torch.Generator,
              patch_size: int = PATCH_SIZE, n_keep: int = N_KEEP) -> tuple[torch.Tensor, torch.Tensor]:
    """A 13-patch view: n_keep positions drawn from the generator, the first m taken from a and the rest from b, each
    at its own position; every other pixel is zero (the null image). Returns (composite, ids_keep (n_keep,))."""
    grid = a.shape[-1] // patch_size
    ids = torch.randperm(grid * grid, generator=generator)[:n_keep]
    out = torch.zeros_like(a)
    for rank, p in enumerate(ids.tolist()):
        r, c = divmod(p, grid)
        window = (..., slice(r * patch_size, (r + 1) * patch_size), slice(c * patch_size, (c + 1) * patch_size))
        out[window] = (a if rank < m else b)[window]
    return out, ids


def both_covered(order: np.ndarray, a_rows: np.ndarray, b_rows: np.ndarray) -> float:
    """Share of pairs whose top-k caption rows (P, k) hold at least one caption of A and one of B."""
    hit_a = (order[:, :, None] == a_rows[:, None, :]).any(axis=(1, 2))
    hit_b = (order[:, :, None] == b_rows[:, None, :]).any(axis=(1, 2))
    return float((hit_a & hit_b).mean())


def balance(scores_a: np.ndarray, scores_b: np.ndarray) -> np.ndarray:
    """A's share of the top 5 when A's and B's 10 captions are ranked by score (ties keep A first)."""
    both = np.concatenate([scores_a, scores_b], axis=1)
    top = np.argsort(-both, axis=1, kind="stable")[:, :5]
    return (top < scores_a.shape[1]).mean(axis=1)


def selection_index(s_blend: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> tuple[np.ndarray, int]:
    """(s_blend - s_b) / (s_a - s_b) per item, 1 when the blend scores like source a and 0 like source b. Items with
    |s_a - s_b| < 1e-6 are dropped; returns the kept indices and the number dropped."""
    s_blend, s_a, s_b = (np.asarray(x, dtype=float) for x in (s_blend, s_a, s_b))
    keep = np.abs(s_a - s_b) >= TIE_TOL
    return (s_blend[keep] - s_b[keep]) / (s_a[keep] - s_b[keep]), int((~keep).sum())
