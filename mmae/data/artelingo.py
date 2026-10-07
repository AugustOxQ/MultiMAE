"""ArtELingo English (WikiArt paintings) for H-b (spec docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md,
section 4): (image, caption, emotion) pairs and 5-caption retrieval sets. Every ArtELingo-28 painting (the dense
human reference) is held out of train and validation; test keeps them."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # some WikiArt scans exceed PIL's decompression-bomb limit; the data is trusted

log = logging.getLogger(__name__)

EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else")
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTIONS)}
CAPTIONS_PER_IMAGE = 5
SPLIT_FILES = {"train": "artelingo_train.json", "val": "artelingo_val.json", "test": "artelingo_test.json"}
RETRIEVAL_FILES = {"val": "artelingo_val_retrieval.json", "test": "artelingo_test_retrieval.json"}
HELDOUT_FILE = Path(__file__).with_name("al28_paintings.txt")
MIN_SIDE = 224  # CLIP's input size: JPEGs are decoded at a reduced scale no smaller than this


def heldout_paintings(path: str | Path | None = None) -> frozenset[str]:
    """The held-out painting names, one per line (None = the packaged ArtELingo-28 list)."""
    text = Path(path or HELDOUT_FILE).read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def read_json(annotations_dir: str | Path, name: str) -> list[dict]:
    with open(Path(annotations_dir) / name, encoding="utf-8") as f:
        return json.load(f)


def load_painting(path: Path) -> Image.Image:
    """RGB image; JPEGs use PIL's draft mode (DCT scaling to the smallest scale with both sides >= MIN_SIDE), which
    makes large WikiArt scans cheap to decode before CLIP's resize to 224."""
    with Image.open(path) as image:
        image.draft("RGB", (MIN_SIDE, MIN_SIDE))
        return image.convert("RGB")


def emotion_lookup(items: list[dict]) -> dict[tuple[str, str], int]:
    """(painting, caption) -> emotion index from a per-caption file; on a duplicate key the first entry wins and
    the number of such keys is logged."""
    lookup: dict[tuple[str, str], int] = {}
    duplicates = 0
    for item in items:
        key = (item["painting"], item["caption"])
        if key in lookup:
            duplicates += lookup[key] != EMOTION_INDEX[item["emotion"]]
            continue
        lookup[key] = EMOTION_INDEX[item["emotion"]]
    if duplicates:
        log.info("%d (painting, caption) keys carry more than one emotion; the first is kept", duplicates)
    return lookup


def retrieval_groups(
    annotations_dir: str | Path, split: str, heldout: frozenset[str] = frozenset(), limit: int | None = None
) -> list[tuple[str, list[str], list[int]]]:
    """(image path, 5 captions, their 5 emotion indices) per painting of the val or test retrieval file, in file
    order. Held-out paintings are dropped from val only (test keeps them); `limit` keeps the first N paintings."""
    if split not in RETRIEVAL_FILES:
        raise ValueError(f"retrieval sets are {sorted(RETRIEVAL_FILES)}, got {split!r}")
    drop = heldout if split == "val" else frozenset()
    lookup = emotion_lookup(read_json(annotations_dir, SPLIT_FILES[split]))
    groups = []
    for item in read_json(annotations_dir, RETRIEVAL_FILES[split]):
        if item["painting"] in drop:
            continue
        captions = item["caption"][:CAPTIONS_PER_IMAGE]
        if len(captions) < CAPTIONS_PER_IMAGE:
            raise ValueError(f"{item['painting']} has {len(captions)} captions, fewer than {CAPTIONS_PER_IMAGE}")
        emotions = []
        for caption in captions:
            key = (item["painting"], caption)
            if key not in lookup:
                raise KeyError(f"no emotion for {key} in {SPLIT_FILES[split]}")
            emotions.append(lookup[key])
        groups.append((item["image"], list(captions), emotions))
    return groups[:limit]


class ArtelingoPairs(Dataset):
    """One (image, caption, emotion index) item per caption. train: every caption of artelingo_train.json except
    the held-out paintings'; val and test: flattened caption-major from the 5-caption retrieval file (an unshuffled
    batch then shows distinct paintings), held-out paintings dropped from val. `limit` keeps the first N pairs."""

    def __init__(
        self, images_dir: str | Path, annotations_dir: str | Path, split: str,
        transform: Callable[[Image.Image], torch.Tensor], limit: int | None = None,
        heldout: frozenset[str] = frozenset(),
    ) -> None:
        if split == "train":
            items = read_json(annotations_dir, SPLIT_FILES["train"])
            self.pairs = [(it["image"], it["caption"], EMOTION_INDEX[it["emotion"]])
                          for it in items if it["painting"] not in heldout]
        else:
            groups = retrieval_groups(annotations_dir, split, heldout)
            self.pairs = [(image, captions[c], emotions[c])
                          for c in range(CAPTIONS_PER_IMAGE) for image, captions, emotions in groups]
        self.pairs = self.pairs[:limit]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        image, caption, emotion = self.pairs[index]
        return self.transform(load_painting(self.images_dir / image)), caption, emotion


class ArtelingoRetrieval(Dataset):
    """One item per painting with its 5 captions (val or test retrieval file); `emotions` holds their labels."""

    def __init__(
        self, images_dir: str | Path, annotations_dir: str | Path, split: str,
        transform: Callable[[Image.Image], torch.Tensor], limit: int | None = None,
        heldout: frozenset[str] = frozenset(),
    ) -> None:
        groups = retrieval_groups(annotations_dir, split, heldout, limit)
        self.items = [(image, captions) for image, captions, _ in groups]
        self.emotions = [emotions for _, _, emotions in groups]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        image, captions = self.items[index]
        return self.transform(load_painting(self.images_dir / image)), list(captions)
