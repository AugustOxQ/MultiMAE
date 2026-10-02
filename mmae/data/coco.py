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


def check_captions(items: list[dict]) -> None:
    short = [item["image"] for item in items if len(item["caption"]) < CAPTIONS_PER_IMAGE]
    if short:
        raise ValueError(f"{len(short)} images have fewer than {CAPTIONS_PER_IMAGE} captions, e.g. {short[0]}")


def retrieval_items(annotations_dir: str | Path, split: str, limit: int | None = None) -> list[tuple[str, list[str]]]:
    """(image path, first 5 captions) per image of the val or test split, in file order; `limit` keeps the
    first N images. CocoRetrieval's items, also used to map the test set to COCO ids (mmae.engine.eccv)."""
    if split == "train":
        raise ValueError("retrieval sets are 'val' and 'test'")
    items = read_split(annotations_dir, split)
    check_captions(items)
    return [(item["image"], item["caption"][:CAPTIONS_PER_IMAGE]) for item in items][:limit]


class CocoPairs(Dataset):
    """One (image, caption) pair per item. train has one caption per entry; val and test are
    flattened from their 5-caption files (no derived *_one_caption.json files needed).

    val and test are caption-major: every image with its first caption, then every image with its
    second caption, and so on. Consecutive pairs (an unshuffled eval batch of at most as many pairs as
    there are images) therefore show distinct images, so no caption is a false negative in the
    contrastive loss. `limit` keeps the first N pairs: for val and test with N up to the number of
    images, that is the first N images with their first caption (the images CocoRetrieval's `limit`
    keeps).
    """

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
            check_captions(items)
            self.pairs = [(item["image"], item["caption"][c]) for c in range(CAPTIONS_PER_IMAGE) for item in items]
        self.pairs = self.pairs[:limit]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image, caption = self.pairs[index]
        return self.transform(load_image(self.images_dir / image)), caption


class CocoRetrieval(Dataset):
    """One item per image with its first 5 captions (Karpathy val/test); `limit` keeps the first N images."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        limit: int | None = None,
    ) -> None:
        self.items = retrieval_items(annotations_dir, split, limit)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        image, captions = self.items[index]
        return self.transform(load_image(self.images_dir / image)), list(captions)
