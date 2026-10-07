"""Dataset selection by data.name: coco (the default when the key is absent, as in every COCO config) or
artelingo (H-b spec 2026-10-07)."""
from __future__ import annotations

from typing import Callable

from omegaconf import DictConfig
from torch.utils.data import Dataset

from mmae.data.artelingo import ArtelingoPairs, ArtelingoRetrieval, heldout_paintings
from mmae.data.coco import CocoPairs, CocoRetrieval

DATASETS = ("coco", "artelingo")


def dataset_name(dcfg: DictConfig) -> str:
    name = str(dcfg.get("name", "coco"))
    if name not in DATASETS:
        raise ValueError(f"data.name must be one of {DATASETS}, got {name!r}")
    return name


def build_pairs(dcfg: DictConfig, split: str, transform: Callable, limit: int | None) -> Dataset:
    if dataset_name(dcfg) == "artelingo":
        return ArtelingoPairs(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit,
                              heldout=heldout_paintings(dcfg.get("heldout_file")))
    return CocoPairs(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit)


def build_retrieval(dcfg: DictConfig, split: str, transform: Callable, limit: int | None) -> Dataset:
    if dataset_name(dcfg) == "artelingo":
        return ArtelingoRetrieval(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit,
                                  heldout=heldout_paintings(dcfg.get("heldout_file")))
    return CocoRetrieval(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit)
