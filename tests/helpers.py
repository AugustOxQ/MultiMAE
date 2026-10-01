"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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


MODEL_NAMES = ["fusion_concat", "fusion_multilearner", "fusion_none", "image_mae", "text_mlm"]
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
