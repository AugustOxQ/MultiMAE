"""Data for the H-b readouts (spec sections 4, 6 and 9): the AL-28 dense targets, English reference counts,
validation labels for temperature fitting, train histograms for the probe and the prior, the caption-length grid,
and the D7 test captions."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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


def al28_targets(csv: str | Path = AL28_CSV, min_votes: int = 20) -> tuple[Paintings, np.ndarray]:
    """Per painting, the 9-class histogram of its non-English AL-28 votes ('other' merged into 'something else');
    paintings with fewer than min_votes such votes are dropped. Sorted by painting name."""
    frame = pd.read_csv(csv, usecols=["painting", "emotion", "language", "image_name"])
    frame = frame[frame.language.str.lower() != "english"]
    frame = frame.assign(emotion=frame.emotion.replace(AL28_MERGE))
    unknown = set(frame.emotion) - set(EMOTION_INDEX)
    if unknown:
        raise ValueError(f"unknown AL-28 labels {sorted(unknown)}")
    items = [{"painting": p, "emotion": e, "image": i} for p, e, i in zip(frame.painting, frame.emotion, frame.image_name)]
    paintings, counts = _histograms(items, lambda _: True)
    keep = counts.sum(1) >= min_votes
    return Paintings([n for n, k in zip(paintings.names, keep) if k], [i for i, k in zip(paintings.images, keep) if k]), counts[keep]


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
