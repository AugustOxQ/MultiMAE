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

# COCO class -> comma-separated caption terms (singular; plurals are generated). Ambiguous generic single words
# (orange, light, glass, plant, stop, mouse, remote) count only inside their phrase ("traffic light", "potted plant").
_CLASS_TERMS = {
    "person": "person,man,woman,boy,girl,child,kid,guy,lady,player",
    "bicycle": "bicycle,bike", "car": "car", "motorcycle": "motorcycle,motorbike",
    "airplane": "airplane,plane,jet", "bus": "bus", "train": "train", "truck": "truck", "boat": "boat",
    "traffic light": "traffic light", "fire hydrant": "fire hydrant", "stop sign": "stop sign",
    "parking meter": "parking meter", "bench": "bench", "bird": "bird", "cat": "cat", "dog": "dog",
    "horse": "horse", "sheep": "sheep", "cow": "cow", "elephant": "elephant", "bear": "bear", "zebra": "zebra",
    "giraffe": "giraffe", "backpack": "backpack", "umbrella": "umbrella", "handbag": "handbag,purse",
    "tie": "tie", "suitcase": "suitcase", "frisbee": "frisbee", "skis": "ski", "snowboard": "snowboard",
    "sports ball": "sports ball,ball", "kite": "kite", "baseball bat": "baseball bat,bat",
    "baseball glove": "baseball glove,glove", "skateboard": "skateboard", "surfboard": "surfboard",
    "tennis racket": "tennis racket,racket,racquet", "bottle": "bottle", "wine glass": "wine glass",
    "cup": "cup", "fork": "fork", "knife": "knife", "spoon": "spoon", "bowl": "bowl", "banana": "banana",
    "apple": "apple", "sandwich": "sandwich", "orange": "", "broccoli": "broccoli", "carrot": "carrot",
    "hot dog": "hot dog,hotdog", "pizza": "pizza", "donut": "donut,doughnut", "cake": "cake", "chair": "chair",
    "couch": "couch,sofa", "potted plant": "potted plant", "bed": "bed", "dining table": "table",
    "toilet": "toilet", "tv": "tv,television", "laptop": "laptop", "mouse": "", "remote": "",
    "keyboard": "keyboard", "cell phone": "cell phone,cellphone,phone", "microwave": "microwave", "oven": "oven",
    "toaster": "toaster", "sink": "sink", "refrigerator": "refrigerator,fridge", "book": "book", "clock": "clock",
    "vase": "vase", "scissors": "scissors", "teddy bear": "teddy bear", "hair drier": "hair drier,hair dryer",
    "toothbrush": "toothbrush",
}
_IRREGULAR = {"man": "men", "woman": "women", "child": "children", "lady": "ladies", "knife": "knives",
              "person": "people", "sheep": "sheep", "scissors": "scissors", "ski": "skis"}


def _plural(term: str) -> str:
    head, _, last = term.rpartition(" ")
    last = _IRREGULAR.get(last) or (last + "es" if last.endswith(("s", "x", "ch", "sh")) else last + "s")
    return f"{head} {last}".strip()


# term (word or phrase, lower case) -> COCO class name
COCO_CLASS_TERMS: dict[str, str] = {}
for _cls, _terms in _CLASS_TERMS.items():
    for _term in filter(None, _terms.split(",")):
        COCO_CLASS_TERMS[_term] = _cls
        COCO_CLASS_TERMS[_plural(_term)] = _cls
_MAX_PHRASE = max(len(t.split()) for t in COCO_CLASS_TERMS)
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
    """Number of distinct COCO classes the caption mentions. Phrases match first (longest first) and consume their
    tokens ("hot dog" is not "dog"); each class counts once however often it is named."""
    tokens = [w.lower() for w in _WORD.findall(caption)]
    found: set[str] = set()
    i = 0
    while i < len(tokens):
        for n in range(min(_MAX_PHRASE, len(tokens) - i), 0, -1):
            cls = COCO_CLASS_TERMS.get(" ".join(tokens[i:i + n]))
            if cls is not None:
                found.add(cls)
                i += n
                break
        else:
            i += 1
    return len(found)


def drop_words(caption: str, kind: str, k: int, rng: random.Random) -> str | None:
    """The caption with k random words of `kind` ("content" or "stop") deleted, or None with fewer than k."""
    words = caption.replace(".", " ").split()
    pick = [i for i, w in enumerate(words) if is_content_word(w) == (kind == "content") and w.isalpha()]
    if len(pick) < k:
        return None
    drop = set(rng.sample(pick, k))
    return " ".join(w for i, w in enumerate(words) if i not in drop)
