"""GPU stage of the H-b readouts (spec sections 6, 8, 9): forward passes only, saved as arrays for the CPU stage.

For each painting: the emotion logits of the caption decoder with the caption fully hidden, for every caption length
of the grid, on V random 25% views and on the full image (decoder models only); the pooled image embedding of the
full image and of each view (every model). For D7: the emotion logits for every test caption with j content tokens
visible (prefix and random subset), on the real image (views or full) and on the null image. The null image uses all
patches even for masked-source models: with no image content the view is irrelevant.

On CUDA the forward passes run under bf16 autocast; logits and embeddings are cast to float32 and saved as float16."""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from mmae.data.artelingo import load_painting
from mmae.data.stopwords import content_token_table
from mmae.models import MultiMAE
from mmae.models.masking import random_patch_mask

NUM_PATCHES, IMAGE_RATIO = 49, 0.75
D7_PATTERNS = ["j0", "prefix1", "prefix2", "prefix4", "prefix8", "random1", "random2", "random4", "random8"]
D7_SEED = 4321


def load_run(run_dir: str | Path, device) -> tuple[MultiMAE, DictConfig]:
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
    state = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


def has_decoder(model: MultiMAE) -> bool:
    return bool(model.reconstruction and model.use_image and model.use_text and model.emotion_head
                and not model.pooled_conditioning)


def _autocast(device):
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _half(x: torch.Tensor) -> torch.Tensor:
    return x.float().half().cpu()


def view_ids(n_items: int, n_views: int, seed: int = 1234) -> torch.Tensor:
    """(n_views, n_items, 13): view v of item i is the same for every model and batch size."""
    views = []
    for v in range(n_views):
        generator = torch.Generator().manual_seed(seed + v)
        ids_keep, _ = random_patch_mask(n_items, NUM_PATCHES, IMAGE_RATIO, generator=generator)
        views.append(ids_keep)
    return torch.stack(views)


def hidden_captions(tokenizer, lengths: list[int], max_text_len: int) -> dict[str, torch.Tensor]:
    """One caption per length with every real token hidden (the filler word is never seen)."""
    enc = tokenizer([" ".join(["the"] * n) for n in lengths], max_length=max_text_len, truncation=True,
                    padding="max_length", return_attention_mask=True, return_special_tokens_mask=True,
                    return_tensors="pt")
    real = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "token_mask": real}


class _Images(Dataset):
    def __init__(self, paths, images_dir, transform):
        self.paths, self.images_dir, self.transform = list(paths), Path(images_dir), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, self.transform(load_painting(self.images_dir / self.paths[i]))


def _loader(paths, images_dir, transform, batch_size, workers=8):
    workers = min(workers, len(paths))
    return DataLoader(_Images(paths, images_dir, transform), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def _emotion_logits(model, images, hidden, ids_keep) -> torch.Tensor:
    """(B, L, 9): every caption length of `hidden` for every image, in one decoder pass over B * L rows
    (image-major: row b * L + l is image b with caption length l)."""
    b, n_len = images.shape[0], hidden["input_ids"].shape[0]
    rep = {k: v.repeat(b, 1) for k, v in hidden.items()}
    ids = None if ids_keep is None else ids_keep.repeat_interleave(n_len, dim=0)
    _, emotion = model.decode_text(images.repeat_interleave(n_len, dim=0), rep["input_ids"], rep["attention_mask"],
                                   rep["token_mask"], ids)
    return emotion.float().reshape(b, n_len, -1)


@torch.no_grad()
def encode_paintings(model, image_paths, images_dir, transform, hidden, views, batch_size, device,
                     workers: int = 8) -> dict[str, np.ndarray]:
    decoder = has_decoder(model)
    hidden = {k: v.to(device) for k, v in hidden.items()}
    keys = ["emb_full", "emb_views"] + (["dec_views", "dec_full"] if decoder else [])
    chunks: dict[str, list] = {k: [] for k in keys}
    for index, images in _loader(image_paths, images_dir, transform, batch_size, workers):
        images = images.to(device)
        batch_views = views[:, index].to(device)  # (V, B, 13)
        with _autocast(device):
            chunks["emb_full"].append(_half(model.vision.pool(model.vision.encode(images))))
            chunks["emb_views"].append(torch.stack(
                [_half(model.vision.pool(model.vision.encode(images, ids))) for ids in batch_views], dim=1))
            if decoder:
                chunks["dec_views"].append(torch.stack(
                    [_half(_emotion_logits(model, images, hidden, ids)) for ids in batch_views], dim=1))
                chunks["dec_full"].append(_half(_emotion_logits(model, images, hidden, None)).unsqueeze(1))
    return {k: torch.cat(v).numpy() for k, v in chunks.items()}


@torch.no_grad()
def encode_embeddings(model, image_paths, images_dir, transform, batch_size, device, workers: int = 8) -> np.ndarray:
    """(N, D) float16 pooled embeddings of the full images."""
    out = []
    for _, images in _loader(image_paths, images_dir, transform, batch_size, workers):
        with _autocast(device):
            out.append(_half(model.vision.pool(model.vision.encode(images.to(device)))))
    return torch.cat(out).numpy()


@torch.no_grad()
def prompt_embeddings(model, tokenizer, max_text_len, device, labels) -> np.ndarray:
    enc = tokenizer([f"a painting that evokes {label}." for label in labels], max_length=max_text_len,
                    truncation=True, padding="max_length", return_tensors="pt")
    with _autocast(device):
        z = model.embed_text(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return z.float().cpu().numpy()


def d7_visible(token_mask_real: torch.Tensor, content: torch.Tensor, pattern: str, generator) -> torch.Tensor | None:
    """(T,) bool of visible positions for one caption, or None if it has fewer content tokens than the pattern needs."""
    positions = torch.nonzero(content & token_mask_real).flatten()
    if pattern == "j0":
        return torch.zeros_like(token_mask_real)
    kind, j = ("prefix", int(pattern[6:])) if pattern.startswith("prefix") else ("random", int(pattern[6:]))
    if len(positions) < j:
        return None
    chosen = positions[:j] if kind == "prefix" else positions[torch.randperm(len(positions), generator=generator)[:j]]
    visible = torch.zeros_like(token_mask_real)
    visible[chosen] = True
    return visible


def _d7_logits(model, images, ids, am, tm, ids_keep) -> torch.Tensor:
    """(B, P, 9): the P visibility patterns of each caption in one pass (row b * P + k); tm is (B, P, T)."""
    b, p = tm.shape[:2]
    keep = None if ids_keep is None else ids_keep.repeat_interleave(p, dim=0)
    _, emotion = model.decode_text(images.repeat_interleave(p, dim=0), ids.repeat_interleave(p, dim=0),
                                   am.repeat_interleave(p, dim=0), tm.reshape(b * p, -1), keep)
    return emotion.float().reshape(b, p, -1)


@torch.no_grad()
def encode_d7(model, captions, images_dir, transform, tokenizer, max_text_len, n_views, batch_size, device,
              workers: int = 8) -> dict[str, np.ndarray]:
    """Emotion logits for every test caption under the 9 visibility patterns (spec section 9)."""
    enc = tokenizer([c["caption"] for c in captions], max_length=max_text_len, truncation=True,
                    padding="max_length", return_attention_mask=True, return_special_tokens_mask=True,
                    return_tensors="pt")
    real = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
    content = content_token_table(tokenizer)[enc["input_ids"]]
    n, p = len(captions), len(D7_PATTERNS)
    token_masks = torch.zeros(n, p, real.shape[1], dtype=torch.bool)
    valid = torch.zeros(n, p, dtype=torch.bool)
    generator = torch.Generator().manual_seed(D7_SEED)
    for i in range(n):
        for k, pattern in enumerate(D7_PATTERNS):
            visible = d7_visible(real[i], content[i], pattern, generator)
            if visible is not None:
                valid[i, k] = True
                token_masks[i, k] = real[i] & ~visible
            else:
                token_masks[i, k] = real[i]
    masked_source = model.mlm_image_source == "masked"
    views = view_ids(n, n_views, seed=5678) if masked_source else None
    real_out, null_out = [], []
    for index, images in _loader([c["image"] for c in captions], images_dir, transform, batch_size, workers):
        images = images.to(device)
        ids, am = enc["input_ids"][index].to(device), enc["attention_mask"][index].to(device)
        tm = token_masks[index].to(device)
        with _autocast(device):
            if masked_source:
                per_view = [_d7_logits(model, images, ids, am, tm, views[v, index].to(device))
                            for v in range(n_views)]
            else:
                per_view = [_d7_logits(model, images, ids, am, tm, None)]
            real_out.append(_half(torch.stack(per_view, dim=2)))  # (B, P, V', 9)
            null_out.append(_half(_d7_logits(model, torch.zeros_like(images), ids, am, tm, None)))
    return {"d7_real": torch.cat(real_out).numpy(), "d7_null": torch.cat(null_out).numpy(),
            "d7_valid": valid.numpy(), "d7_label": np.array([c["emotion"] for c in captions], dtype=np.int64)}
