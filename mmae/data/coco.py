"""COCO Karpathy splits: image-caption pairs (training and loss evaluation) and 5-caption retrieval sets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few COCO files are truncated
CAPTIONS_PER_IMAGE = 5  # some images have 6 captions; like v0, we use the first 5
SPLIT_FILES = {
    "train": "coco_karpathy_train.json",
    "val": "coco_karpathy_val.json",
    "test": "coco_karpathy_test.json",
}


def read_split(annotations_dir: str | Path, split: str) -> list[dict]:
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(SPLIT_FILES)}")
    with open(Path(annotations_dir) / SPLIT_FILES[split], encoding="utf-8") as f:
        return json.load(f)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class CocoPairs(Dataset):
    """One (image, caption) pair per item. train has one caption per entry; val and test are
    flattened from their 5-caption files, so no derived *_one_caption.json files are needed."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        limit: int | None = None,
    ) -> None:
        items = read_split(annotations_dir, split)
        if split == "train":
            self.pairs = [(item["image"], item["caption"]) for item in items]
        else:
            self.pairs = [(item["image"], c) for item in items for c in item["caption"][:CAPTIONS_PER_IMAGE]]
        self.pairs = self.pairs[:limit]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image, caption = self.pairs[index]
        return self.transform(load_image(self.images_dir / image)), caption


class CocoRetrieval(Dataset):
    """One item per image with its first 5 captions (Karpathy val/test)."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        limit: int | None = None,
    ) -> None:
        if split == "train":
            raise ValueError("retrieval sets are 'val' and 'test'")
        items = read_split(annotations_dir, split)
        short = [item["image"] for item in items if len(item["caption"]) < CAPTIONS_PER_IMAGE]
        if short:
            raise ValueError(f"{len(short)} images have fewer than {CAPTIONS_PER_IMAGE} captions, e.g. {short[0]}")
        self.items = [(item["image"], item["caption"][:CAPTIONS_PER_IMAGE]) for item in items][:limit]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        image, captions = self.items[index]
        return self.transform(load_image(self.images_dir / image)), list(captions)
