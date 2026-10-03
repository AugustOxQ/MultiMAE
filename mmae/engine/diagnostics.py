"""Stage 0 diagnostics (spec 2026-10-03, section 5) on COCO 5k test embeddings.

Class sets come from the PMRP ground truth (pmrp_ground_truth, zeta = 0): a caption's t2i list holds every test
image whose COCO object-class set equals that of the caption's image, so images whose lists coincide share a set.
"""
from __future__ import annotations

import random
import re

import numpy as np
import torch

from mmae.data.stopwords import is_content_word
from mmae.engine.eccv import pmrp_ground_truth

# The 80 COCO category names as caption words (singular and plural), with common person words.
COCO_CLASS_WORDS = frozenset("""
person persons people man men woman women boy boys girl girls child children kid kids guy guys lady ladies player
players bicycle bicycles bike bikes car cars motorcycle motorcycles motorbike airplane airplanes plane planes jet bus
buses train trains truck trucks boat boats traffic light lights fire hydrant hydrants stop sign signs parking meter
meters bench benches bird birds cat cats dog dogs horse horses sheep cow cows elephant elephants bear bears zebra
zebras giraffe giraffes backpack backpacks umbrella umbrellas handbag handbags purse tie ties suitcase suitcases
frisbee frisbees skis ski snowboard snowboards ball balls kite kites bat bats glove gloves skateboard skateboards
surfboard surfboards racket rackets racquet bottle bottles glass glasses cup cups fork forks knife knives spoon
spoons bowl bowls banana bananas apple apples sandwich sandwiches orange oranges broccoli carrot carrots hotdog
pizza pizzas donut donuts doughnut doughnuts cake cakes chair chairs couch couches sofa plant plants bed beds table
tables toilet toilets tv tvs television laptop laptops computer mouse remote remotes keyboard keyboards phone phones
microwave microwaves oven ovens toaster sink sinks refrigerator refrigerators fridge book books clock clocks vase
vases scissors teddy toothbrush toothbrushes
""".split())
_WORD = re.compile(r"[A-Za-z']+")


def class_set_groups(pm_gt: dict, image_ids: np.ndarray, caption_ids: np.ndarray) -> np.ndarray:
    gt = pmrp_ground_truth(pm_gt, image_ids, caption_ids)
    owner = dict(zip(caption_ids.reshape(-1).tolist(), np.repeat(image_ids, caption_ids.shape[1]).tolist()))
    row = {image: i for i, image in enumerate(image_ids.tolist())}
    keys: dict[tuple[int, ...], int] = {}
    groups = np.full(len(image_ids), -1, dtype=np.int64)
    for caption, images in gt["t2i"].items():
        group = keys.setdefault(tuple(images), len(keys))
        groups[row[owner[caption]]] = group
    return groups


@torch.no_grad()
def neighbour_purity(emb: torch.Tensor, groups: np.ndarray, k: int = 10, owners: np.ndarray | None = None,
                     chunk: int = 1024) -> float:
    """Percent of each query's top-k most similar other items (cosine) in its own group, averaged over the queries
    with a group (>= 0) that has another member outside the query's owner. owners (n,): items with the query's
    owner are never neighbours (e.g. the other captions of a caption's image)."""
    n = emb.shape[0]
    g = torch.as_tensor(groups)
    own = torch.as_tensor(owners if owners is not None else np.arange(n))
    shares = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = emb[start:stop] @ emb.T
        scores[own[start:stop, None] == own[None, :]] = -float("inf")  # self and same-owner items
        top = scores.topk(k, dim=1).indices
        qg = g[start:stop]
        mates = ((g[None, :] == qg[:, None]) & (own[None, :] != own[start:stop, None])).sum(dim=1)
        valid = (qg >= 0) & (mates > 0)
        hit = (g[top] == qg[:, None]).double().mean(dim=1)
        shares.append(hit[valid])
    return 100.0 * torch.cat(shares).mean().item()


def pmrp_rows(pm_gt: dict, image_ids: np.ndarray, caption_ids: np.ndarray) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    gt = pmrp_ground_truth(pm_gt, image_ids, caption_ids)
    image_row = {image: i for i, image in enumerate(image_ids.tolist())}
    caption_row = {c: i for i, c in enumerate(caption_ids.reshape(-1).tolist())}
    t2i_q = sorted(gt["t2i"], key=caption_row.__getitem__)
    i2t_q = sorted(gt["i2t"], key=image_row.__getitem__)
    return {
        "t2i": (np.array([caption_row[c] for c in t2i_q]), [np.array([image_row[i] for i in gt["t2i"][c]]) for c in t2i_q]),
        "i2t": (np.array([image_row[i] for i in i2t_q]), [np.array([caption_row[c] for c in gt["i2t"][i]]) for i in i2t_q]),
    }


@torch.no_grad()
def per_query_rprecision(query_emb: torch.Tensor, cand_emb: torch.Tensor, positives: list[np.ndarray],
                         max_r: int = 50, chunk: int = 1024) -> np.ndarray:
    """R-Precision per query, R = min(#positives, max_r). Ties break by torch.topk, so the mean can differ from the
    package's PMRP (stable sort) by a few hundredths."""
    out = np.zeros(len(positives))
    depth = min(max_r, cand_emb.shape[0])
    for start in range(0, len(positives), chunk):
        stop = min(start + chunk, len(positives))
        top = (query_emb[start:stop] @ cand_emb.T).topk(depth, dim=1).indices.cpu().numpy()
        for j in range(stop - start):
            pos = positives[start + j]
            r = min(len(pos), max_r)
            out[start + j] = np.isin(top[j, :r], pos).sum() / r
    return out


@torch.no_grad()
def similarity_stats(image_emb: torch.Tensor, caption_emb: torch.Tensor, groups: np.ndarray, chunk: int = 512) -> dict[str, float]:
    """Mean image-caption cosine of own pairs, of other pairs whose images share a class set, and of the rest."""
    n, k, d = caption_emb.shape
    text = caption_emb.reshape(n * k, d)
    text_group = torch.as_tensor(np.repeat(groups, k))
    text_owner = torch.arange(n).repeat_interleave(k)
    g = torch.as_tensor(groups)
    sums = {"pos": 0.0, "same_class_neg": 0.0, "other_neg": 0.0}
    counts = dict.fromkeys(sums, 0)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = (image_emb[start:stop] @ text.T).double()
        own = text_owner[None, :] == torch.arange(start, stop)[:, None]
        same = (text_group[None, :] == g[start:stop, None]) & (g[start:stop, None] >= 0) & ~own
        other = ~own & ~same
        for key, mask in (("pos", own), ("same_class_neg", same), ("other_neg", other)):
            sums[key] += scores[mask].sum().item()
            counts[key] += int(mask.sum())
    return {key: sums[key] / counts[key] if counts[key] else float("nan") for key in sums}


def count_class_words(caption: str) -> int:
    return sum(word.lower() in COCO_CLASS_WORDS for word in _WORD.findall(caption))


def drop_words(caption: str, kind: str, k: int, rng: random.Random) -> str | None:
    """The caption with k random words of `kind` ("content" or "stop") deleted, or None with fewer than k."""
    words = caption.replace(".", " ").split()
    pick = [i for i, w in enumerate(words) if is_content_word(w) == (kind == "content") and w.isalpha()]
    if len(pick) < k:
        return None
    drop = set(rng.sample(pick, k))
    return " ".join(w for i, w in enumerate(words) if i not in drop)
