"""SemEval-2023 Task 1, Visual Word Sense Disambiguation (VWSD; Raganato et al., 2023; data CC-BY-NC 4.0).

Each item is a possibly ambiguous target word, a short phrase that fixes its sense ("football goal") and ten
candidate images. The model ranks the candidates by cosine similarity between the phrase's text embedding and the
image embeddings; Hit@1 and MRR are reported in percent (rank = 1 + the number of candidates scored strictly
higher, as in mmae.engine.retrieval). Layout of `root`, the released test package: {lang}.test.data*.txt
(target <tab> phrase <tab> 10 image names), {lang}.test.gold*.txt (the gold image name per line) and
test_images_resized/ with the images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGES_SUBDIR = "test_images_resized"


@dataclass(frozen=True)
class VwsdItem:
    word: str
    phrase: str
    candidates: tuple[str, ...]
    gold: str


def _one_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {root}, found {[m.name for m in matches]}")
    return matches[0]


def read_vwsd(root: str | Path, lang: str = "en") -> list[VwsdItem]:
    root = Path(root)
    lines = [l for l in _one_file(root, f"{lang}.test.data*.txt").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    gold = [l.strip() for l in _one_file(root, f"{lang}.test.gold*.txt").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if len(lines) != len(gold):
        raise ValueError(f"{lang} VWSD: {len(lines)} items but {len(gold)} gold labels")
    items = []
    for number, (line, answer) in enumerate(zip(lines, gold), start=1):
        fields = [f.strip() for f in line.split("\t")]
        candidates = tuple(f for f in fields[2:] if f)
        if answer not in candidates:
            raise ValueError(f"{lang} VWSD line {number}: gold image {answer!r} is not one of its candidates")
        items.append(VwsdItem(fields[0], fields[1], candidates, answer))
    return items


class VwsdImages(Dataset):
    def __init__(self, root: str | Path, names: list[str], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.folder = Path(root) / IMAGES_SUBDIR
        self.names = names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.folder / self.names[i]) as image:
            return self.transform(image.convert("RGB"))  # PNGs with alpha or palettes, grayscale JPEGs


def vwsd_metrics(image_emb: torch.Tensor, text_emb: torch.Tensor, items: list[VwsdItem], index: dict[str, int]) -> dict[str, float]:
    """image_emb (M, D) rows indexed by image name (index), text_emb (Q, D) one row per item."""
    ranks = []
    for row, item in enumerate(items):
        scores = image_emb[[index[name] for name in item.candidates]] @ text_emb[row]
        gold = scores[item.candidates.index(item.gold)]
        ranks.append(1 + int((scores > gold).sum()))
    ranks_t = torch.tensor(ranks, dtype=torch.double)
    return {
        "vwsd/hit1": 100.0 * (ranks_t == 1).double().mean().item(),
        "vwsd/mrr": 100.0 * (1.0 / ranks_t).mean().item(),
        "vwsd/n": float(len(items)),
    }


@torch.no_grad()
def evaluate_vwsd(
    model, root: str | Path, transform, collator, lang: str = "en", prompt: str = "{phrase}",
    batch_size: int = 256, num_workers: int = 0, device: torch.device | str | None = None,
) -> dict[str, float]:
    """Hit@1 and MRR of `model` (an unwrapped MultiMAE: embed_image / embed_text) on the VWSD test set."""
    items = read_vwsd(root, lang)
    names = sorted({name for item in items for name in item.candidates})
    index = {name: i for i, name in enumerate(names)}
    device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    loader = DataLoader(VwsdImages(root, names, transform), batch_size=batch_size, num_workers=num_workers)
    image_emb = torch.cat([model.embed_image(images.to(device)).float() for images in loader])
    queries = [prompt.format(phrase=item.phrase, word=item.word) for item in items]
    text_emb = []
    for start in range(0, len(queries), batch_size):
        tokens = collator.tokenize(queries[start : start + batch_size])
        text_emb.append(model.embed_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device)).float())
    model.train(was_training)
    return vwsd_metrics(image_emb.cpu(), torch.cat(text_emb).cpu(), items, index)
