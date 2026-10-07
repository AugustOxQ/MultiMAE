"""Data for the H-b readouts (spec sections 4, 6 and 9): the AL-28 dense targets, English reference counts,
validation labels for temperature fitting, train histograms for the probe and the prior, the caption-length grid,
and the D7 test captions."""
from __future__ import annotations

import csv as csvlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mmae.data.artelingo import EMOTION_INDEX, SPLIT_FILES, read_json

AL28_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"
AL28_MERGE = {"other": "something else"}


@dataclass
class Paintings:
    names: list[str]
    images: list[str]


def _histograms(items: list[dict], keep) -> tuple[Paintings, np.ndarray]:
    counts: dict[str, np.ndarray] = {}
    images: dict[str, str] = {}
    for item in items:
        if not keep(item["painting"]):
            continue
        row = counts.setdefault(item["painting"], np.zeros(9, dtype=np.int64))
        row[EMOTION_INDEX[item["emotion"]]] += 1
        images.setdefault(item["painting"], item["image"])
    names = sorted(counts)
    return Paintings(names, [images[n] for n in names]), np.stack([counts[n] for n in names]) if names else np.zeros((0, 9))


def _al28_items(csv: str | Path) -> list[dict]:
    """The non-English AL-28 votes as items; 'other' is merged into 'something else' and flagged."""
    with open(csv, newline="", encoding="utf-8") as handle:
        rows = [r for r in csvlib.DictReader(handle) if r["language"].lower() != "english"]
    other = [r["emotion"] == "other" for r in rows]
    emotions = [AL28_MERGE.get(r["emotion"], r["emotion"]) for r in rows]
    unknown = set(emotions) - set(EMOTION_INDEX)
    if unknown:
        raise ValueError(f"unknown AL-28 labels {sorted(unknown)}")
    return [{"painting": r["painting"], "emotion": e, "image": r["image_name"], "other": o}
            for r, e, o in zip(rows, emotions, other)]


def al28_targets(csv: str | Path = AL28_CSV, min_votes: int = 20, drop_other: bool = False) -> tuple[Paintings, np.ndarray]:
    """Per painting, the 9-class histogram of its non-English AL-28 votes ('other' merged into 'something else', or
    removed when drop_other); paintings with fewer than min_votes such votes (counted with 'other') are dropped, so
    both variants hold the same paintings. Sorted by painting name."""
    items = _al28_items(csv)
    paintings, counts = _histograms(items, lambda _: True)
    keep = counts.sum(1) >= min_votes
    names = [n for n, k in zip(paintings.names, keep) if k]
    images = [i for i, k in zip(paintings.images, keep) if k]
    counts = counts[keep]
    if drop_other:
        kept = set(names)
        sub = [it for it in items if it["painting"] in kept and not it["other"]]
        sub_paintings, counts = _histograms(sub, lambda _: True)
        if sub_paintings.names != names:
            raise ValueError("some AL-28 paintings have only 'other' votes")
    return Paintings(names, images), counts


def al28_votes(csv: str | Path = AL28_CSV, min_votes: int = 20) -> dict[str, np.ndarray]:
    """Per painting (same filter as al28_targets), the label index of each non-English vote, 'other' merged."""
    votes: dict[str, list[int]] = defaultdict(list)
    for it in _al28_items(csv):
        votes[it["painting"]].append(EMOTION_INDEX[it["emotion"]])
    return {p: np.array(v, dtype=np.int64) for p, v in sorted(votes.items()) if len(v) >= min_votes}


def english_counts(names: list[str], annotations_dir: str | Path) -> np.ndarray:
    """The ArtELingo English label counts of each named painting, from the train, val and test files (zeros when
    a painting has none)."""
    wanted = set(names)
    totals = {n: np.zeros(9, dtype=np.int64) for n in names}
    for split in ("train", "val", "test"):
        for item in read_json(annotations_dir, SPLIT_FILES[split]):
            if item["painting"] in wanted:
                totals[item["painting"]][EMOTION_INDEX[item["emotion"]]] += 1
    return np.stack([totals[n] for n in names])


def val_labels(annotations_dir: str | Path, heldout: frozenset[str]) -> tuple[Paintings, np.ndarray, np.ndarray]:
    """Validation paintings (held-out ones excluded) and every individual label with its painting's index."""
    items = [it for it in read_json(annotations_dir, SPLIT_FILES["val"]) if it["painting"] not in heldout]
    paintings, _ = _histograms(items, lambda _: True)
    position = {n: i for i, n in enumerate(paintings.names)}
    index = np.array([position[it["painting"]] for it in items], dtype=np.int64)
    labels = np.array([EMOTION_INDEX[it["emotion"]] for it in items], dtype=np.int64)
    return paintings, index, labels


def train_histograms(annotations_dir: str | Path, heldout: frozenset[str]) -> tuple[Paintings, np.ndarray]:
    return _histograms(read_json(annotations_dir, SPLIT_FILES["train"]), lambda p: p not in heldout)


def prior(annotations_dir: str | Path, heldout: frozenset[str]) -> np.ndarray:
    _, counts = train_histograms(annotations_dir, heldout)
    total = counts.sum(0).astype(np.float64)
    return total / total.sum()


def length_grid(annotations_dir: str | Path, tokenizer, heldout: frozenset[str], max_text_len: int) -> list[int]:
    """Caption lengths (real, non-special tokens) at the 10th, ..., 90th percentiles of ArtELingo train, truncated
    as the collator truncates (max_text_len - 2)."""
    captions = [it["caption"] for it in read_json(annotations_dir, SPLIT_FILES["train"]) if it["painting"] not in heldout]
    ids = tokenizer(captions, add_special_tokens=False)["input_ids"]
    lengths = np.minimum([len(x) for x in ids], max_text_len - 2)
    grid = np.percentile(lengths, np.arange(10, 100, 10), method="nearest")
    return [max(1, int(x)) for x in grid]


def test_captions(annotations_dir: str | Path) -> list[dict]:
    """Every caption of artelingo_test.json with its annotator's emotion index (D7)."""
    return [{"painting": it["painting"], "image": it["image"], "caption": it["caption"],
             "emotion": EMOTION_INDEX[it["emotion"]]} for it in read_json(annotations_dir, SPLIT_FILES["test"])]
