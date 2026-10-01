"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "configs")
CLIP_NAME = "openai/clip-vit-base-patch32"
