"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "configs")
CLIP_NAME = "openai/clip-vit-base-patch32"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_ddp_worker(command: str, out_dir: Path, nproc: int = 2, *args: str) -> subprocess.CompletedProcess:
    """Run tests/_ddp_workers.py <command> under torchrun with `nproc` CPU processes (gloo)."""
    env = {**os.environ, "ACCELERATE_USE_CPU": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
        "--master_port", str(free_port()), str(REPO / "tests" / "_ddp_workers.py"),
        command, "--out", str(out_dir), *args,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)


MODEL_NAMES = ["fusion_concat", "fusion_multilearner", "fusion_none", "contrastive", "image_mae", "text_mlm"]
CAPTIONS = [
    "a dog running on the beach",
    "two people riding horses",
    "a red bus parked next to a tall building in the city",
    "a cat",
]


def compose_cfg(*overrides: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def make_batch(tokenizer, batch_size: int = 4, max_len: int = 32, captions: list[str] | None = None) -> dict:
    """Smooth random images (memorizable under norm_pix) and real CLIP-tokenized captions."""
    captions = captions or [CAPTIONS[i % len(CAPTIONS)] for i in range(batch_size)]
    enc = tokenizer(
        captions, max_length=max_len, truncation=True, padding="max_length",
        return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt",
    )
    g = torch.Generator().manual_seed(0)
    coarse = torch.randn(batch_size, 3, 7, 7, generator=g)
    images = F.interpolate(coarse, size=224, mode="bilinear", align_corners=False)
    return {
        "pixel_values": images,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "special_tokens_mask": enc["special_tokens_mask"],
    }


def add_content_mask(batch: dict, tokenizer) -> dict:
    """The content_tokens_mask a Collator(content_words=True) would add to a make_batch() batch."""
    from mmae.data.stopwords import content_token_table

    table = content_token_table(tokenizer)
    content = table[batch["input_ids"]] & batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    return {**batch, "content_tokens_mask": content}


def make_fake_coco(root: Path) -> tuple[Path, Path]:
    """A tiny COCO-like tree: 8 train images (2 captions each), 6 val and 6 test images.

    Includes the oddities real COCO has: a grayscale image, a CMYK image, a truncated JPEG and an
    image with 6 captions.
    """
    images_dir, annotations_dir = root / "images", root / "annotations"
    (images_dir / "train").mkdir(parents=True)
    (images_dir / "val").mkdir()
    annotations_dir.mkdir()
    g = torch.Generator().manual_seed(0)

    def save(rel: str, mode: str = "RGB") -> str:
        pixels = (torch.rand(3, 120, 160, generator=g) * 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(pixels, "RGB").convert(mode).save(images_dir / rel, "JPEG")
        return rel

    train = []
    for i in range(8):
        rel = save(f"train/{i}.jpg", "L" if i == 1 else ("CMYK" if i == 2 else "RGB"))
        train += [{"image": rel, "caption": f"train caption {i} {k}", "image_id": i} for k in range(2)]
    truncated = images_dir / "train" / "3.jpg"
    truncated.write_bytes(truncated.read_bytes()[: int(truncated.stat().st_size * 0.7)])

    def split(name: str) -> list[dict]:
        items = []
        for i in range(6):
            rel = save(f"val/{name}_{i}.jpg")
            n = 6 if i == 0 else 5
            items.append({"image": rel, "caption": [f"{name} {i} caption {k}" for k in range(n)]})
        return items

    for name, items in (("train", train), ("val", split("val")), ("test", split("test"))):
        (annotations_dir / f"coco_karpathy_{name}.json").write_text(json.dumps(items))
    return images_dir, annotations_dir


def make_fake_vwsd(root: Path) -> Path:
    """A tiny VWSD test package: 3 English items over 12 images (JPEG RGB, PNG RGBA, grayscale JPEG)."""
    images = root / "test_images_resized"
    images.mkdir(parents=True)
    g = torch.Generator().manual_seed(1)
    names = []
    for i in range(12):
        name = f"image.{i}." + ("png" if i % 3 == 1 else "jpg")
        pixels = (torch.rand(3, 64, 80, generator=g) * 255).byte().permute(1, 2, 0).numpy()
        img = Image.fromarray(pixels, "RGB")
        if i % 3 == 1:
            img.convert("RGBA").save(images / name, "PNG")
        else:
            img.convert("L" if i % 3 == 2 else "RGB").save(images / name, "JPEG")
        names.append(name)
    rows = [("goal", "football goal", names[0:10]), ("seat", "eating seat", names[2:12]), ("bank", "river bank", names[1:11])]
    (root / "en.test.data.v1.1.txt").write_text("".join(f"{w}\t{p}\t" + "\t".join(c) + "\n" for w, p, c in rows))
    (root / "en.test.gold.v1.1.txt").write_text(f"{names[3]}\n{names[11]}\n{names[1]}\n")
    return root


def run_train(cwd: Path, fake_coco, *overrides: str, nproc: int = 1, script: str = "train.py") -> subprocess.CompletedProcess:
    """Run train.py (or evaluate.py) on the fake COCO with the tiny CLIP on CPU, from `cwd`.

    Extended test metrics are off explicitly (they need the real COCO test split); tests may override it.
    """
    images_dir, annotations_dir = fake_coco
    args = [
        f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
        "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
        f"paths.res_dir={cwd / 'res'}", "eval.extended_metrics=false", *overrides,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "1", "WANDB_MODE": "disabled"}
    if nproc == 1:
        cmd = [sys.executable, str(REPO / script), *args]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
               "--master_port", str(free_port()), str(REPO / script), *args]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1200)


def run_train_artelingo(cwd: Path, fake_artelingo, *overrides: str, script: str = "train.py") -> subprocess.CompletedProcess:
    """Run train.py (or evaluate.py) on the fake ArtELingo with the tiny CLIP on CPU, from `cwd`."""
    images_dir, annotations_dir, heldout = fake_artelingo
    args = [
        "data=artelingo", f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
        f"data.heldout_file={heldout}", "model.backbone.pretrained=tiny-random-clip", "train=debug",
        "train.num_workers=0", f"paths.res_dir={cwd / 'res'}", *overrides,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "1", "WANDB_MODE": "disabled"}
    return subprocess.run([sys.executable, str(REPO / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=1200)


def add_emotion(batch: dict) -> dict:
    """An ArtELingo-style emotion label per caption (class index in 0..8)."""
    b = batch["input_ids"].shape[0]
    return {**batch, "emotion": torch.arange(b) % 9}


ARTELINGO_EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else")


def make_fake_artelingo(root: Path) -> tuple[Path, Path, Path]:
    """A tiny ArtELingo tree: 6 train paintings (2 captions each, p5 held out), 3 val and 3 test paintings with 5
    captions each (v2 and t1 held out), per-caption files with emotions and 5-caption retrieval files, a held-out
    list, one grayscale and one PNG-as-RGBA painting. Returns (images_dir, annotations_dir, heldout_file)."""
    images_dir, annotations_dir = root / "wikiart", root / "artelingo"
    (images_dir / "Style_A").mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)

    def save(name: str, mode: str = "RGB") -> str:
        rel = f"Style_A/{name}.jpg"
        pixels = (torch.rand(3, 300, 260, generator=g) * 255).byte().permute(1, 2, 0).numpy()
        Image.fromarray(pixels, "RGB").convert(mode).save(images_dir / rel, "JPEG")
        return rel

    def caption_items(painting: str, rel: str, n: int, offset: int) -> list[dict]:
        return [{"image": rel, "caption": f"{painting} reading {k} of the painting", "image_id": f"{painting}#{k}",
                 "emotion": ARTELINGO_EMOTIONS[(offset + k) % 9], "art_style": "Style_A", "painting": painting}
                for k in range(n)]

    train = []
    for i in range(6):
        rel = save(f"p{i}", "L" if i == 1 else "RGB")
        train += caption_items(f"p{i}", rel, 2, i)
    files = {"artelingo_train.json": train}
    for split, prefix in (("val", "v"), ("test", "t")):
        per_caption, retrieval = [], []
        for i in range(3):
            painting = f"{prefix}{i}"
            rel = save(painting)
            items = caption_items(painting, rel, 5, i)
            per_caption += items
            retrieval.append({"image": rel, "caption": [x["caption"] for x in items], "image_id": painting,
                              "art_style": "Style_A", "painting": painting})
        files[f"artelingo_{split}.json"] = per_caption
        files[f"artelingo_{split}_retrieval.json"] = retrieval
    for name, items in files.items():
        (annotations_dir / name).write_text(json.dumps(items))
    heldout = root / "al28_paintings.txt"
    heldout.write_text("p5\nv2\nt1\nnot_in_artelingo\n")
    return images_dir, annotations_dir, heldout
