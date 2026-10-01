"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""
import os
import socket
import subprocess
import sys
from pathlib import Path

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
