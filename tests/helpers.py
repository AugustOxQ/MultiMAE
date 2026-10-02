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
