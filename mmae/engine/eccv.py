"""Extended COCO 5k test metrics: ECCV Caption, CxC, COCO 1K/5K recalls and PMRP.

The metric functions are those of the `eccv_caption` package (naver-ai/eccv-caption, Chun et al., "ECCV
Caption", ECCV 2022; MIT). This module maps our test set to COCO ids and builds the ranked lists the package
reads, cut to what each metric looks at, so no 25,000 x 5,000 ranking is turned into Python lists:

- COCO 5K and CxC R@{1,5,10} read the first 10 items of a query's list. ECCV R@1, R-Precision and mAP@R
  read the first R items, R = the query's number of ECCV positives (at most 48 i2t, 19 t2i). One list of
  the top `depth` items per query (depth >= 10, every R, and the PMRP cut) serves all of them.
- COCO 1K R@K keeps the items of a query's list that are in its fold (the package's 5 folds of
  coco_test_ids: 5,000 captions and their 1,000 images each), then reads the first K. Each query is
  evaluated in exactly one fold (checked), so a second list per query, its top 10 among its own fold's
  candidates, gives the package the same filtered prefix as the full ranking would. The paper's numbers
  come from 5K top-50 lists filtered to the fold instead, which can leave fewer than 10 fold items and
  under-counts its R@5 and R@10 (zero-shot CLIP ViT-B/32 i2t/t2i R@10: 95.00/87.70 there, 95.68/88.74
  here; R@1 agrees), so only COCO 1K R@1 is comparable with the paper.
- PMRP is the ECCV Caption paper's modified PMRP (Sec. 5, Table 4): plausible matches at zeta = 0 and
  R = min(#PM positives, 50). The authors' example passes top-50 lists to the package, whose R-Precision
  then reads min(R, 50) items; we pass the same 50-item prefixes. The released PM files leave out each
  query's own pair (the authors' data_tools/plausible_matching_func.py with omit_orig=True): no t2i list
  holds the caption's image, no i2t list the image's captions, so the 1,130 captions of the 226 images
  whose COCO class set no other test image has get empty lists and those images have no i2t list. The
  paper scores with the own pairs in (the function's default, omit_orig=False), and so do we:
  pmrp_ground_truth adds them back, and every caption the PM files list (24,760) and every image of those
  captions (4,952) is a query. Zero-shot CLIP ViT-B/32 then gives 55.31 (i2t 59.95, t2i 50.68) against the
  paper's 55.32; the released lists alone, empty ones left out, gave 51.11.

Rankings sort by similarity with ties in dataset order (a stable sort), so every list is a prefix of the
full ranking and each fold list is the full ranking restricted to the fold. mmae.engine.retrieval's
retrieval_metrics counts ties optimistically instead (rank = 1 + the number of candidates scored strictly
higher), so coco5k/* can differ slightly from the plain retrieval metrics when two embeddings tie exactly.
Zero-shot CLIP ViT-B/32 has one such tie: the own caption "A group of chefs preparing food inside of a
kitchen." of the test image at position 2357 is also a caption of the image at position 1892, which comes
first, so coco5k/i2t_r1 is 50.10 and retrieval i2t_R1 50.12.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from eccv_caption import Metrics
from omegaconf import DictConfig

from mmae.data.coco import retrieval_items

log = logging.getLogger(__name__)

COCO_5K_IMAGES = 5000
CAPTIONS_FILE = "captions_val2014.json"  # in data.annotations_dir; the Karpathy test images are all val2014
PM_FILES = ("pm_image_to_caption.json", "pm_caption_to_image.json")  # in data.pm_dir
RECALL_KS = (1, 5, 10)
PMRP_MAX_R = 50
DIRECTIONS = ("i2t", "t2i")
NUM_FOLDS = 5
_IMAGE_ID = re.compile(r"(\d+)\.\w+$")


def coco_image_id(path: str) -> int:
    """val2014/COCO_val2014_000000391895.jpg -> 391895."""
    match = _IMAGE_ID.search(Path(path).name)
    if match is None:
        raise ValueError(f"no COCO image id in the file name {path!r}")
    return int(match.group(1))


def map_coco_ids(items: Sequence[tuple[str, Sequence[str]]], captions_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """COCO image ids (N,) and caption annotation ids (N, K) for retrieval items (image path, K captions).

    A caption gets the id of the COCO annotation of its image with exactly the same text. When an image
    lists one text twice (three test images do, each copy a COCO annotation of its own), the copies take
    the ids in annotation-file order. Raises ValueError for a caption with no such annotation left.
    """
    with open(captions_file, encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    by_image: dict[int, dict[str, list[int]]] = {}
    for ann in annotations:
        by_image.setdefault(int(ann["image_id"]), {}).setdefault(ann["caption"], []).append(int(ann["id"]))
    image_ids, caption_ids = [], []
    for image, captions in items:
        image_id = coco_image_id(image)
        unused = {text: list(ids) for text, ids in by_image.get(image_id, {}).items()}
        row = []
        for text in captions:
            if not unused.get(text):
                raise ValueError(
                    f"caption {text!r} of {image} matches no unused annotation of COCO image {image_id} in {captions_file}"
                )
            row.append(unused[text].pop(0))
        image_ids.append(image_id)
        caption_ids.append(row)
    return np.array(image_ids, dtype=np.int64), np.array(caption_ids, dtype=np.int64)


def check_ids(image_ids: np.ndarray, caption_ids: np.ndarray, metrics: Metrics) -> None:
    """Raise ValueError unless the mapped test set is the package's: the same caption ids as its
    coco_test_ids, each caption under the image the package's COCO ground truth gives it."""
    flat = caption_ids.reshape(-1).tolist()
    expected = {int(c) for c in metrics.coco_ids}
    if len(set(flat)) != len(flat) or set(flat) != expected:
        raise ValueError(
            f"mapped caption ids ({len(set(flat))} distinct of {len(flat)}) differ from the package's "
            f"{len(expected)} COCO test ids: {len(set(flat) - expected)} extra, {len(expected - set(flat))} missing"
        )
    if len(set(image_ids.tolist())) != len(image_ids):
        raise ValueError("an image appears more than once in the test set")
    owners = np.repeat(image_ids, caption_ids.shape[1]).tolist()
    wrong = [c for c, i in zip(flat, owners) if metrics.coco_gts["t2i"].get(c) != [i]]
    if wrong:
        raise ValueError(f"{len(wrong)} captions belong to another image in the package's COCO ground truth, e.g. {wrong[0]}")


def coco_1k_folds(metrics: Metrics, image_ids: np.ndarray, caption_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold of every image (N,) and caption (N*K,), as Metrics.coco_1k_recalls defines the folds: fold f
    holds coco_ids[f*M:(f+1)*M] (M = len(coco_ids) // 5) and their images. Needs check_ids to have passed."""
    size = len(metrics.coco_ids) // NUM_FOLDS
    if size * NUM_FOLDS != len(metrics.coco_ids):
        raise ValueError(f"{len(metrics.coco_ids)} COCO test ids do not split into {NUM_FOLDS} folds")
    fold_of = {int(c): i // size for i, c in enumerate(metrics.coco_ids)}
    caption_fold = np.array([fold_of[c] for c in caption_ids.reshape(-1).tolist()], dtype=np.int64)
    per_image = caption_fold.reshape(caption_ids.shape)
    split = np.nonzero((per_image != per_image[:, :1]).any(axis=1))[0]
    if len(split):
        raise ValueError(f"{len(split)} images have captions in two COCO 1K folds, e.g. {image_ids[split[0]]}")
    return per_image[:, 0].copy(), caption_fold


def pmrp_ground_truth(
    pm_gts: dict[str, dict[int, list[int]]], image_ids: np.ndarray, caption_ids: np.ndarray
) -> dict[str, dict[int, list[int]]]:
    """The PMRP queries and positives of the ECCV Caption paper: the PM files' lists plus each query's own pair.

    Queries are every caption with a t2i list (an empty one included) and every image of those captions
    (one with no i2t list included). A caption's positives are its t2i list and its own image, an image's
    its i2t list and its own captions, so no list is empty. Captions the PM files do not list (their image
    has no COCO object annotation) are not queries. image_ids (N,) and caption_ids (N, K) are the test
    set's COCO ids (map_coco_ids). Raises ValueError for a t2i query that is not a test caption or an i2t
    query that is not the image of a t2i query.
    """
    owner = dict(zip(caption_ids.reshape(-1).tolist(), np.repeat(image_ids, caption_ids.shape[1]).tolist()))
    own = dict(zip(image_ids.tolist(), caption_ids.tolist()))
    outside = [c for c in pm_gts["t2i"] if c not in owner]
    if outside:
        raise ValueError(f"{len(outside)} PM t2i queries are not test captions, e.g. {outside[0]}")
    t2i = {c: sorted(set(p) | {owner[c]}) for c, p in pm_gts["t2i"].items()}
    images = {owner[c] for c in t2i}
    stray = [i for i in pm_gts["i2t"] if i not in images]
    if stray:
        raise ValueError(f"{len(stray)} PM i2t queries are not images of PM t2i queries, e.g. {stray[0]}")
    i2t = {i: sorted(set(pm_gts["i2t"].get(i, ())) | set(own[i])) for i in image_ids.tolist() if i in images}
    assert all(t2i.values()) and all(i2t.values()), "an empty PMRP list: the package divides by zero on it"
    return {"i2t": i2t, "t2i": t2i}


def _rank(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    candidate_ids: np.ndarray,
    depth: int,
    query_fold: np.ndarray,
    candidate_fold: np.ndarray,
    chunk: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Per query, the ids of its top `depth` candidates and of the top 10 candidates of its own fold."""
    ids = np.array(candidate_ids.tolist(), dtype=object)  # Python ints, shared by every list
    depth = min(depth, len(candidates))
    device = candidates.device
    fold_columns = {
        int(f): torch.as_tensor(np.nonzero(candidate_fold == f)[0], device=device) for f in np.unique(query_fold)
    }
    top: list[list[int]] = []
    fold_top: list[list[int]] = [[] for _ in range(len(queries))]
    for start in range(0, len(queries), chunk):
        scores = queries[start : start + chunk] @ candidates.T
        order = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :depth]
        top.extend(ids[order.cpu().numpy()].tolist())
        folds = query_fold[start : start + chunk]
        for f in np.unique(folds):
            rows = np.nonzero(folds == f)[0]
            columns = fold_columns[int(f)]
            sub = scores[torch.as_tensor(rows, device=device)][:, columns]
            best = torch.sort(sub, dim=1, descending=True, stable=True).indices[:, : max(RECALL_KS)]
            for row, items in zip(rows.tolist(), ids[columns[best].cpu().numpy()].tolist()):
                fold_top[start + row] = items
    return top, fold_top


@torch.no_grad()
def coco_test_metrics(
    image_emb: torch.Tensor,
    caption_emb: torch.Tensor,
    image_ids: np.ndarray,
    caption_ids: np.ndarray,
    metrics: Metrics,
    pmrp_max_r: int = PMRP_MAX_R,
    chunk: int = 256,
) -> dict[str, float]:
    """ECCV Caption, CxC, COCO 1K/5K and (if `metrics` has PM ground truth) PMRP, in percent.

    image_emb (N, D) and caption_emb (N, K, D) in dataset order (encode_retrieval_set's output);
    image_ids (N,) and caption_ids (N, K) their COCO ids (map_coco_ids). Keys: eccv/{i2t,t2i}_{map_at_r,
    rprecision,r1}, {cxc,coco1k,coco5k}/{i2t,t2i}_r{1,5,10}, pmrp/{i2t,t2i}, the i2t/t2i means the ECCV
    Caption paper's Table 4 reports (eccv/map_at_r, eccv/rprecision, eccv/r1, cxc/r1, coco1k/r1,
    coco5k/r1, pmrp) and coco1k/rsum, the six COCO 1K recalls summed from exact full rankings within each
    fold. That is higher than the paper's RSUM, whose R@5 and R@10 come from 5K top-50 lists filtered to
    the fold, and not comparable with it. Raises ValueError unless the ids are the package's test set
    (check_ids).
    """
    n, k, dim = caption_emb.shape
    if image_emb.shape != (n, dim) or image_ids.shape != (n,) or caption_ids.shape != (n, k):
        raise ValueError(
            f"embeddings {tuple(image_emb.shape)}, {tuple(caption_emb.shape)} do not match ids "
            f"{image_ids.shape}, {caption_ids.shape}"
        )
    check_ids(image_ids, caption_ids, metrics)
    image_fold, caption_fold = coco_1k_folds(metrics, image_ids, caption_ids)
    flat_caption_ids = caption_ids.reshape(-1)
    pm = bool(metrics.pm_gts)
    depth = {
        d: max([max(RECALL_KS), *(len(p) for p in metrics.eccv_gts[d].values())] + ([pmrp_max_r] if pm else []))
        for d in DIRECTIONS
    }
    text_emb = caption_emb.reshape(n * k, dim).float()
    image_emb = image_emb.float()
    i2t, i2t_fold = _rank(image_emb, text_emb, flat_caption_ids, depth["i2t"], image_fold, caption_fold, chunk)
    t2i, t2i_fold = _rank(text_emb, image_emb, image_ids, depth["t2i"], caption_fold, image_fold, chunk)
    image_keys, caption_keys = image_ids.tolist(), flat_caption_ids.tolist()
    top = {"i2t": dict(zip(image_keys, i2t)), "t2i": dict(zip(caption_keys, t2i))}

    # Asking for eccv_map_at_r is what makes compute_all_metrics return eccv_rprecision too (it tests a
    # misspelt 'eccv_rpresion'), so all three are requested and checked below.
    scores = metrics.compute_all_metrics(
        top["i2t"], top["t2i"], target_metrics=("coco_5k_recalls", "cxc_recalls", "eccv_map_at_r", "eccv_rprecision", "eccv_r1"),
        Ks=RECALL_KS,
    )
    scores.update(metrics.compute_all_metrics(
        dict(zip(image_keys, i2t_fold)), dict(zip(caption_keys, t2i_fold)), target_metrics=("coco_1k_recalls",), Ks=RECALL_KS,
    ))
    expected = {f"{name}_r{K}" for name in ("coco_5k", "cxc", "coco_1k") for K in RECALL_KS}
    expected |= {"eccv_map_at_r", "eccv_rprecision", "eccv_r1"}
    if expected - set(scores):
        raise RuntimeError(f"eccv_caption did not return {sorted(expected - set(scores))}")
    if pm:
        # The package's pmrp reads self.pm_gts: a shallow copy carries the ground truth with the own pairs.
        pm_metrics = copy.copy(metrics)
        pm_metrics.pm_gts = pmrp_ground_truth(metrics.pm_gts, image_ids, caption_ids)
        retrieved = {d: {q: top[d][q][:pmrp_max_r] for q in pm_metrics.pm_gts[d]} for d in DIRECTIONS}
        scores["pmrp"] = pm_metrics.pmrp(retrieved, "all")
    return _flatten(scores)


def _flatten(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in DIRECTIONS:
        for name in ("map_at_r", "rprecision", "r1"):
            out[f"eccv/{d}_{name}"] = 100.0 * float(scores[f"eccv_{name}"][d])
        for prefix, key in (("cxc", "cxc"), ("coco1k", "coco_1k"), ("coco5k", "coco_5k")):
            for K in RECALL_KS:
                out[f"{prefix}/{d}_r{K}"] = 100.0 * float(scores[f"{key}_r{K}"][d])
        if "pmrp" in scores:
            out[f"pmrp/{d}"] = 100.0 * float(scores["pmrp"][d])
    for prefix, name in (("eccv", "map_at_r"), ("eccv", "rprecision"), ("eccv", "r1"), ("cxc", "r1"), ("coco1k", "r1"), ("coco5k", "r1")):
        out[f"{prefix}/{name}"] = (out[f"{prefix}/i2t_{name}"] + out[f"{prefix}/t2i_{name}"]) / 2
    out["coco1k/rsum"] = sum(out[f"coco1k/{d}_r{K}"] for d in DIRECTIONS for K in RECALL_KS)
    if "pmrp" in scores:
        out["pmrp"] = (out["pmrp/i2t"] + out["pmrp/t2i"]) / 2
    return out


class CocoExtendedMetrics:
    """The extended metrics of the COCO 5k test split. Construction maps and checks the ids and finds the PM
    files (fast, so a misconfigured run fails at start-up); calling it loads the ground truth (about 1 GB
    with PM) and scores the embeddings."""

    def __init__(self, items: Sequence[tuple[str, Sequence[str]]], annotations_dir: str | Path, pm_dir: str | Path | None) -> None:
        self.image_ids, self.caption_ids = map_coco_ids(items, Path(annotations_dir) / CAPTIONS_FILE)
        check_ids(self.image_ids, self.caption_ids, Metrics())
        self.pm_dir = Path(pm_dir) if pm_dir else None
        if self.pm_dir is None:
            log.info("data.pm_dir is null: PMRP is skipped")
        else:
            missing = [name for name in PM_FILES if not (self.pm_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"PMRP ground truth {missing} not in data.pm_dir={self.pm_dir} (null skips PMRP)")

    def __call__(self, image_emb: torch.Tensor, caption_emb: torch.Tensor) -> dict[str, float]:
        metrics = Metrics(extra_file_dir=str(self.pm_dir) if self.pm_dir else None)
        return coco_test_metrics(image_emb, caption_emb, self.image_ids, self.caption_ids, metrics)


def build_extended_metrics(cfg: DictConfig, split: str, two_modalities: bool) -> CocoExtendedMetrics | None:
    """The extended metrics for an evaluation of `split`, or None when they do not apply: eval.extended_metrics
    off, not the test split, a single-modality model (silently), data.limit_test set or a test split that is
    not COCO's 5,000 images (one info line)."""
    if not cfg.eval.extended_metrics or split != "test" or not two_modalities:
        return None
    reason = None
    if cfg.data.limit_test is not None:
        reason = f"data.limit_test={cfg.data.limit_test}; they need the full test split"
    else:
        items = retrieval_items(cfg.data.annotations_dir, split)
        if len(items) != COCO_5K_IMAGES:
            reason = f"the test split has {len(items)} images, not COCO's {COCO_5K_IMAGES}"
    if reason is not None:
        log.info("extended test metrics (ECCV Caption, CxC, COCO 1K, PMRP) skipped: %s", reason)
        return None
    return CocoExtendedMetrics(items, cfg.data.annotations_dir, cfg.data.pm_dir)
