# MultiMAE v1 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v0 code (`src/`, `main_*.py`) with a correct, small, extendable `mmae` package: a CLIP-based fusion masked autoencoder trained with Accelerate, configured with Hydra, logged to wandb and to one `res/` folder per run.

**Architecture:** One model class (`MultiMAE`) built from HF CLIP towers with real input masking (MAE token dropping for images, a learned `[MASK]` embedding for text), a pluggable fusion module (`none | concat | multilearner`) and two query decoders. `forward(batch)` returns every loss, so DDP needs no special handling. A plain training loop (`Trainer`) does per-epoch validation with COCO retrieval, early stopping on `val/retrieval/rsum`, best-weight restore and a final test.

**Tech Stack:** Python 3.11, torch 2.11.0 (cu130), torchvision 0.26.0, transformers 5.6.2, accelerate 1.13.0, hydra-core 1.3.2, omegaconf 2.3.0, wandb 0.30.0, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-mmae-refactor-design.md` (read it before starting any task).

## Global Constraints

- Python env: `/root/miniconda3/envs/MultiMAE/bin/python` (written `$PY` below). Run everything from the repo root `/project/MultiAlign/MultiMAE`. Never install into another conda env. Never `pip install` anything not listed in `requirements.txt` or `extras_require["dev"]`.
- Branch: `refactor`. One commit per task. Commit with the repo author identity and the session trailers:
  ```bash
  git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -m "<message>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
  ```
  Never push, never rewrite history, never touch the `main` or `legacy` branches or the `legacy-v0` tag.
- Package name `mmae`. Nothing imports `src.*`. No timm, no `datasets`, no PyTorch Lightning, no einops.
- Library code logs with `logging`, never `print`. Comments and docstrings in English. Type hints on public functions.
- Pinned versions (from the `MultiMAE` env): torch==2.11.0, torchvision==0.26.0, transformers==5.6.2, accelerate==1.13.0, hydra-core==1.3.2, omegaconf==2.3.0, wandb==0.30.0, pillow==12.3.0, tqdm==4.70.1, matplotlib==3.11.2, numpy==2.2.6.
- Fast tests: `$PY -m pytest` (slow tests are deselected by `pytest.ini`) must pass at the end of every task. Slow tests (`$PY -m pytest -m slow`) need the real CLIP checkpoint, COCO at `/data/SSD/coco/` or a GPU; run the ones your task adds.
- One RTX 3090 is shared with other jobs. Fast tests must not need a GPU.
- Debug work (scratch scripts, outputs) goes in `tests/20261001_<name>/` with a log `20261001_<name>_log.md` (problem, steps, root cause, fix). Edits to files that existed before this plan (`setup.py`, `requirements.txt`, `README.md`, `.gitignore`) get an entry in `.claude/<yyyymmdd>_log.md` (per-file `# <path>` header, before/after snippet, why).
- Do not edit `docs/superpowers/specs/`. Do not delete `res/`, `data/`, `wandb/` or `outputs/` contents.
- Learning rates (`train.lr` 1e-4, `train.lr_backbone` 1e-5) are untested placeholders; do not tune them.

## Review Focus

1. A caption with no maskable token (empty caption `""` tokenizes to BOS EOS only) in a training batch: no token of that row is masked, the MLM loss stays finite, the batch still trains. Pinned by tests in Task 5 (`test_mlm_loss_with_no_masked_tokens_is_zero_with_grad`) and Task 6 (`test_empty_caption_in_batch_is_finite`).
2. Mixed precision (`train.precision: bf16`): forward under autocast returns finite float32 losses. Pinned in Task 6 (`test_bf16_autocast_losses_are_finite_float32`).
3. Real COCO oddities: grayscale and CMYK images, truncated JPEGs, images with 6 captions. Each loads as a `(3, 224, 224)` tensor, and only the first 5 captions are used. Pinned in Task 7 (`test_odd_images_load_as_rgb`, `test_val_flattens_first_five_captions`).
4. Two runs started in the same second with the same name (parallel seeds): each gets its own folder. Pinned in Task 9 (`test_same_second_same_name_gets_distinct_folders`).
5. A run that crashes mid-training: `run.json` says `failed`, `error.txt` holds the traceback, the exception still propagates. Pinned in Task 9 (`test_crash_marks_run_failed_and_reraises`).

---

## File structure

```
setup.py, requirements.txt, pytest.ini, README.md     # Task 1 (README in Task 12)
train.py                                              # Task 10
evaluate.py                                           # Task 11
configs/config.yaml, model/{base,fusion_concat,fusion_multilearner,fusion_none,image_mae,text_mlm}.yaml,
        data/coco.yaml, train/default.yaml            # Task 6
configs/train/debug.yaml, data/coco_cluster.yaml      # Task 10
mmae/__init__.py                                      # Task 1
mmae/models/masking.py                                # Task 2
mmae/models/backbones.py                              # Task 3
mmae/models/fusion.py, decoders.py                    # Task 4
mmae/losses.py                                        # Task 5
mmae/models/model.py, models/__init__.py              # Task 6
mmae/data/{transforms,coco,collate,__init__}.py       # Task 7
mmae/engine/retrieval.py                              # Task 8
mmae/utils/{run,logging}.py, scripts/list_runs.py     # Task 9
mmae/engine/trainer.py, engine/__init__.py            # Task 10
scripts/run_local.sh, run_cluster.sh                  # Task 12
scripts/check_reports_sum.py, docs/reports/...        # Task 1 (+ report in Task 12)
tests/conftest.py, tests/helpers.py                   # Task 1, extended in Tasks 6, 7
tests/test_*.py, tests/_ddp_workers.py, tests/reference_v0_retrieval.py
```

The spec's layout lists `mmae/utils/seed.py` and `dist.py`; `accelerate.utils.set_seed` and the `Accelerator` cover both jobs, so they are not created.

---

### Task 1: Scaffold the package and docs, remove v0

**Files:**
- Delete: `src/`, `main_fusion_mmae.py`, `main_mae.py`, `main_mlm.py`, `main_mmae.py`, `configs/fusion_mmae_config.yaml`, `configs/mae_config.yaml`, `configs/mlm_config.yaml`, `configs/mmae_config.yaml`, `scripts/check_accelerate.sh`, `scripts/run_fusion_mmae.sh`, `scripts/run_fusion_mmae_cluster.sh`, `notebook/`, `tests/.gitkeep`, `docs/.gitkeep`
- Move: `docs/templates/` → `docs/archive/templates/`
- Modify: `setup.py`, `requirements.txt`
- Create: `pytest.ini`, `mmae/__init__.py`, `mmae/models/__init__.py`, `mmae/data/__init__.py`, `mmae/engine/__init__.py`, `mmae/utils/__init__.py`, `tests/conftest.py`, `tests/helpers.py`, `tests/test_package.py`, `scripts/check_reports_sum.py`, `docs/reports/reports_sum.md`, `docs/reports/auto/v0/pilots/20261001_fusion_ddp_fixes/20261001_fusion_ddp_fixes_log.md`, `docs/reports/auto/v0/pilots/20261001_legacy_entrypoints/20261001_legacy_entrypoints_log.md`

**Interfaces:**
- Produces: importable package `mmae` (`mmae.__version__ == "1.0.0"`), empty subpackages `mmae.models`, `mmae.data`, `mmae.engine`, `mmae.utils`; `tests/helpers.py` with `REPO`, `CONFIG_DIR`, `CLIP_NAME`; pytest marker `slow`.

- [ ] **Step 1: Remove the v0 tree and move the templates**

```bash
git rm -r -q src main_fusion_mmae.py main_mae.py main_mlm.py main_mmae.py \
  configs/fusion_mmae_config.yaml configs/mae_config.yaml configs/mlm_config.yaml configs/mmae_config.yaml \
  scripts/check_accelerate.sh scripts/run_fusion_mmae.sh scripts/run_fusion_mmae_cluster.sh \
  notebook tests/.gitkeep docs/.gitkeep
mkdir -p docs/archive && git mv docs/templates docs/archive/templates
rm -rf src  # leftover untracked __pycache__ / ignored outputs inside src/, if any
git status --short | head -50
```
Expected: only deletions and the rename are staged; `ls` shows no `src/`.

- [ ] **Step 1b: Anchor the `.gitignore` folder patterns to the repo root**

The v0 `.gitignore` has a bare `data/` pattern, which also matches `mmae/data/` and `configs/data/`, so `git add` would silently skip both. In the `# Folder` section change these lines (and only these):
```
data/      ->  /data/
other/     ->  /other/
out/       ->  /out/
output/    ->  /output/
outputs/   ->  /outputs/
notebooks/ ->  /notebooks/
res/       ->  /res/
```
Verify: `mkdir -p mmae/data configs/data && touch mmae/data/x.py configs/data/x.yaml && git check-ignore -v mmae/data/x.py configs/data/x.yaml; echo "exit=$?"; rm mmae/data/x.py configs/data/x.yaml`
Expected: no output from `git check-ignore` and `exit=1` (not ignored). `git check-ignore -v data/cifar` still reports `/data/`.

- [ ] **Step 2: Copy the v0 debug logs into the reports layout (docs only)**

```bash
mkdir -p docs/reports/auto/v0/pilots/20261001_fusion_ddp_fixes docs/reports/auto/v0/pilots/20261001_legacy_entrypoints
git show legacy-v0:src/test/20261001_fusion_ddp_fixes/20261001_fusion_ddp_fixes_log.md \
  > docs/reports/auto/v0/pilots/20261001_fusion_ddp_fixes/20261001_fusion_ddp_fixes_log.md
git show legacy-v0:src/test/20261001_legacy_entrypoints/20261001_legacy_entrypoints_log.md \
  > docs/reports/auto/v0/pilots/20261001_legacy_entrypoints/20261001_legacy_entrypoints_log.md
```

- [ ] **Step 3: Write the failing package test**

`tests/test_package.py`:
```python
import mmae


def test_package_imports():
    assert mmae.__version__ == "1.0.0"
```

`tests/helpers.py`:
```python
"""Shared test helpers (importable because pytest puts tests/ on sys.path)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "configs")
CLIP_NAME = "openai/clip-vit-base-patch32"
```

`tests/conftest.py`:
```python
"""Pytest fixtures shared by all tests."""
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -m "not slow"
markers =
    slow: needs the real CLIP checkpoint, COCO on disk or a GPU; run with `pytest -m slow`
```

Run: `$PY -m pytest tests/test_package.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'mmae'`; pytest itself may also be missing, then install it in Step 5 first and re-run).

- [ ] **Step 4: Create the package and packaging files**

`mmae/__init__.py`:
```python
"""MultiMAE: a fusion masked autoencoder on CLIP towers."""

__version__ = "1.0.0"
```

`mmae/models/__init__.py`, `mmae/data/__init__.py`, `mmae/engine/__init__.py`, `mmae/utils/__init__.py`: each a one-line docstring, e.g. `"""Model components: towers, masking, fusion, decoders and the MultiMAE model."""`, `"""COCO datasets, preprocessing and batch collation."""`, `"""Training loop and retrieval evaluation."""`, `"""Run folders and logging."""`.

`setup.py` (replace the whole file):
```python
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(name: str) -> list[str]:
    lines = (Path(__file__).parent / name).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


setup(
    name="mmae",
    version="1.0.0",
    description="Multimodal masked autoencoder (fusion MAE) on CLIP towers",
    packages=find_packages(include=["mmae", "mmae.*"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": ["pytest>=8"]},
    python_requires=">=3.10",
)
```

`requirements.txt` (replace the whole file):
```
# torch/torchvision: install the CUDA 13.0 build first:
#   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision==0.26.0
transformers==5.6.2
accelerate==1.13.0
hydra-core==1.3.2
omegaconf==2.3.0
wandb==0.30.0
pillow==12.3.0
tqdm==4.70.1
matplotlib==3.11.2
numpy==2.2.6
```

- [ ] **Step 5: Reinstall the package editable**

```bash
$PY -m pip uninstall -y multi-mae
$PY -m pip install --no-deps -e ".[dev]" && $PY -m pip install "pytest>=8"
$PY -c "import mmae, sys; print(mmae.__file__)"
```
Expected: prints `/project/MultiAlign/MultiMAE/mmae/__init__.py`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `$PY -m pytest -q`
Expected: `1 passed`.

- [ ] **Step 7: Port the reports index checker and write the index**

Copy CoSiR's checker unchanged (it resolves `docs/reports` from its own location):
```bash
cp /project/CoSiR/scripts/check_reports_sum.py scripts/check_reports_sum.py
```

`docs/reports/reports_sum.md`:
```markdown
# Reports guide

This file is the index to every report in `docs/reports/`. Keep it current: every new report gets one row here (see [Adding a report](#adding-a-report)). All dates are 2026 unless marked otherwise.

## Layout

| Folder | What goes there |
|---|---|
| `auto/<line>/` | **Automatic reports.** One per experiment, diagnostic, review or brainstorm, written when the task finishes. Grouped by research line: `v0`, `v1`. |
| `auto/<line>/pilots/<dir>/` | Pilot reports and debug logs copied from `tests/<dir>/` (or v0's `src/test/<dir>/`) on branches or tags whose code is not on main. |
| `stage/` | **Stage reports.** Syntheses across several experiments, including progress and comprehensive reports, plus their slide markdown. |
| `weekly/` | **Weekly reports** and their slide markdown. |
| `pptx/` | **Slide decks (.pptx).** Built from slide markdown by `assets/build_*_slides.py`. `*.pptx` is gitignored. |
| `assets/` | Figures, figure data and build scripts used by the reports. |

Research lines:
- **v0:** the pre-refactor code (`src/`, `main_*.py`), frozen at tag `legacy-v0` and branch `legacy`.
- **v1:** the refactored `mmae` package, from 2026-10-01. This is what main contains.

## Start here

- **Design of v1:** the [refactor spec](../superpowers/specs/2026-10-01-mmae-refactor-design.md) and its [implementation plan](../superpowers/plans/2026-10-01-mmae-refactor.md).

## Adding a report

1. **Name:** `YYYY-MM-DD_<topic>.md`. Leave out words the folder already says: no `weekly_`, `stage_report`, `_report` or `mmae_`.
2. **Place:** put it in `auto/<line>/`, `stage/` or `weekly/`. Slide markdown sits next to its report. Decks go in `pptx/`, their build script in `assets/build_<date>_<topic>_slides.py`.
3. **Index:** add one row to the matching table below, oldest first, with a one-line description. A new research line gets `auto/<line>/`, a table here and a row in the list above.
4. **Reports on other branches:** these stay on their branch until a stage report or comparison needs them. Then copy the docs (never the code, never with `git merge`) into this layout and add rows here.
5. **Check:** run `python scripts/check_reports_sum.py`. It fails if any report, deck or pilot folder is missing here, if a link here is broken, or if a file sits loose in `docs/reports/`.

## auto/v0: pre-refactor code

| Date | Report | What it is |
|---|---|---|
| 10-01 | [20261001_legacy_entrypoints](auto/v0/pilots/20261001_legacy_entrypoints/) | Debug log: v0 entry points imported the hook submodule instead of the function; wandb-off crash, CIFAR path and NaN-blind test fixes |
| 10-01 | [20261001_fusion_ddp_fixes](auto/v0/pilots/20261001_fusion_ddp_fixes/) | Debug log: multi-GPU (DDP) fixes, per-batch retrieval gather and best-weight restore in the v0 fusion hooks, with the before/after harness |

## auto/v1: refactored code

| Date | Report | What it is |
|---|---|---|
```

Run: `$PY scripts/check_reports_sum.py`
Expected: `reports_sum.md: OK`.

- [ ] **Step 8: Change log and commit**

Append to `.claude/<yyyymmdd>_log.md` (create if missing) a `# setup.py`, a `# requirements.txt` and a `# .gitignore` section: before/after snippets (package `src` → `mmae`, name `multi-mae` → `mmae`, unpinned list → pinned list without timm/datasets/einops/sklearn/pandas/seaborn/jupyter; folder patterns anchored to the root) and why (v1 refactor, spec §3; `data/` would have ignored `mmae/data/`).

```bash
git add -A .gitignore setup.py requirements.txt pytest.ini mmae tests scripts/check_reports_sum.py docs
git status --short
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Scaffold the mmae package and docs layout; remove the v0 tree" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 2: Masking

**Files:**
- Create: `mmae/models/masking.py`
- Test: `tests/test_masking.py`

**Interfaces:**
- Produces:
  - `random_patch_mask(batch_size: int, num_patches: int, ratio: float, device: torch.device | str | None = None, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]` returning `ids_keep` `(B, N_keep)` long (ascending visible indices) and `mask` `(B, N)` bool (True = masked).
  - `random_token_mask(attention_mask: torch.Tensor, special_tokens_mask: torch.Tensor, ratio: float, generator: torch.Generator | None = None) -> torch.Tensor` returning `(B, T)` bool (True = masked).

- [ ] **Step 1: Write the failing tests**

`tests/test_masking.py`:
```python
import pytest
import torch

from mmae.models.masking import random_patch_mask, random_token_mask


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_patch_mask_counts_and_complement():
    ids_keep, mask = random_patch_mask(4, 49, 0.75, generator=gen())
    assert ids_keep.shape == (4, 13) and mask.shape == (4, 49)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [36, 36, 36, 36]
    for b in range(4):
        assert torch.equal(ids_keep[b], (~mask[b]).nonzero().squeeze(1))


def test_patch_mask_differs_across_samples_and_is_reproducible():
    a = random_patch_mask(8, 49, 0.75, generator=gen(1))[1]
    b = random_patch_mask(8, 49, 0.75, generator=gen(1))[1]
    assert torch.equal(a, b)
    assert len({tuple(row.tolist()) for row in a}) == 8


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_patch_mask_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError):
        random_patch_mask(2, 49, ratio)


def token_batch():
    # CLIP layout: BOS words EOS PAD... where PAD == EOS and padding is flagged special.
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
    special = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    return attention, special


def test_token_mask_counts_and_never_special_or_padding():
    attention, special = token_batch()
    mask = random_token_mask(attention, special, 0.5, generator=gen())
    assert mask.dtype == torch.bool and mask.shape == attention.shape
    assert mask.sum(dim=1).tolist() == [2, 1, 0]  # round(0.5*4)=2, max(1, round(0.5))=1, nothing maskable
    assert not (mask & special.bool()).any()
    assert not (mask & ~attention.bool()).any()


def test_token_mask_masks_at_least_one_token():
    attention, special = token_batch()
    mask = random_token_mask(attention, special, 0.15, generator=gen())
    assert mask.sum(dim=1).tolist() == [1, 1, 0]


def test_token_mask_differs_across_samples_and_is_reproducible():
    attention = torch.tensor([[1] * 12] * 32)
    special = torch.tensor([[1] + [0] * 10 + [1]] * 32)
    a = random_token_mask(attention, special, 0.3, generator=gen(3))
    b = random_token_mask(attention, special, 0.3, generator=gen(3))
    assert torch.equal(a, b)
    assert len({tuple(row.tolist()) for row in a}) > 1


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_token_mask_rejects_bad_ratio(ratio):
    attention, special = token_batch()
    with pytest.raises(ValueError):
        random_token_mask(attention, special, ratio)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_masking.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.models.masking'`.

- [ ] **Step 3: Implement**

`mmae/models/masking.py`:
```python
"""Random masks: MAE-style patch dropping for images and MLM-style token masking for text."""
from __future__ import annotations

import torch


def _uniform(shape: tuple[int, ...], device, generator: torch.Generator | None) -> torch.Tensor:
    """Uniform noise drawn on the generator's device (if any), then moved to `device`."""
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator, device=generator.device).to(device)


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    ratio: float,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask `int(num_patches * ratio)` random patches per sample (argsort of noise, as in MAE).

    Returns:
        ids_keep: (B, N_keep) long, indices of the visible patches in ascending order.
        mask: (B, N) bool, True where the patch is masked.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"patch mask ratio must be in [0, 1), got {ratio}")
    num_keep = num_patches - int(num_patches * ratio)
    noise = _uniform((batch_size, num_patches), device, generator)
    ids_keep = noise.argsort(dim=1)[:, :num_keep].sort(dim=1).values
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=noise.device)
    mask.scatter_(1, ids_keep, False)
    return ids_keep, mask


def random_token_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask `max(1, round(ratio * n))` of the n real, non-special tokens of each caption.

    Captions with no maskable token get no mask. Returns (B, T) bool, True where masked.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"token mask ratio must be in (0, 1), got {ratio}")
    maskable = attention_mask.bool() & ~special_tokens_mask.bool()
    n = maskable.sum(dim=1)
    k = torch.clamp(torch.round(n.float() * ratio), min=1).long()
    k = torch.where(n > 0, torch.minimum(k, n), torch.zeros_like(k))
    noise = _uniform(tuple(maskable.shape), maskable.device, generator)
    noise = noise.masked_fill(~maskable, 2.0)  # non-maskable positions sort after every maskable one
    ranks = noise.argsort(dim=1).argsort(dim=1)
    return ranks < k.unsqueeze(1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_masking.py -q`
Expected: all pass. Then `$PY -m pytest -q` (all fast tests) passes.

- [ ] **Step 5: Confirm the guards bite, then commit**

Temporarily change `noise.masked_fill(~maskable, 2.0)` to `noise.masked_fill(~maskable, 0.0)`, run `test_token_mask_counts_and_never_special_or_padding`: it must FAIL. Restore. Temporarily drop `.sort(dim=1).values` from `ids_keep`: `test_patch_mask_counts_and_complement` must FAIL. Restore.

```bash
git add mmae/models/masking.py tests/test_masking.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add patch and token masking" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 3: CLIP towers

**Files:**
- Create: `mmae/models/backbones.py`
- Test: `tests/test_backbones.py`
- Modify: `tests/conftest.py` (add `tokenizer` fixture)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (all in `mmae.models.backbones`):
  - `TINY_CLIP = "tiny-random-clip"`; `tiny_clip_config() -> CLIPConfig`; `load_clip(pretrained: str) -> CLIPModel`.
  - `class ClipVisionTower(nn.Module)`: `__init__(clip: CLIPModel, pooling: str = "native")`; attributes `hidden_size`, `patch_size`, `num_patches`; `encode(pixel_values, ids_keep=None) -> (B, 1+N_keep, hidden)`; `embed(pixel_values) -> (B, proj)` L2-normalized; `pretrained_parameters()`; `native_head_parameters()`.
  - `class ClipTextTower(nn.Module)`: `__init__(clip, pooling="native")`; attributes `hidden_size`, `vocab_size`, `max_positions`, `mask_embedding` (Parameter, shape `(hidden,)`); `encode(input_ids, attention_mask, token_mask=None) -> (B, T, hidden)`; `embed(input_ids, attention_mask) -> (B, proj)`; `pretrained_parameters()`; `native_head_parameters()`.
  - `class Towers(NamedTuple)`: `vision`, `text`, `logit_scale`.
  - `build_backbone(kind: str, pretrained: str, pooling: str = "native") -> Towers`; registry `BACKBONES = {"hf_clip": build_hf_clip}`.

- [ ] **Step 1: Add the tokenizer fixture**

`tests/conftest.py` (replace):
```python
"""Pytest fixtures shared by all tests."""
import pytest

from helpers import CLIP_NAME


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CLIP_NAME)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_backbones.py`:
```python
import pytest
import torch
import torch.nn.functional as F

from helpers import CLIP_NAME
from mmae.models.backbones import (
    TINY_CLIP,
    ClipTextTower,
    ClipVisionTower,
    build_backbone,
    load_clip,
)

CAPTIONS = ["a photo of a cat on a mat", "dog", "two people riding horses on the beach today"]


def towers_for(name: str):
    torch.manual_seed(0)
    clip = load_clip(name).eval()
    return clip, ClipVisionTower(clip).eval(), ClipTextTower(clip).eval()


def text_inputs(tokenizer, max_len: int = 16):
    enc = tokenizer(CAPTIONS, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def pixels(batch: int = 3, size: int = 224) -> torch.Tensor:
    return torch.randn(batch, 3, size, size, generator=torch.Generator().manual_seed(1))


def check_equivalence(clip, vision, text, tokenizer):
    px = pixels()
    ids, am = text_inputs(tokenizer)
    with torch.no_grad():
        # encode() without a mask equals HF's own hidden states
        assert torch.allclose(vision.encode(px), clip.vision_model(pixel_values=px).last_hidden_state, atol=1e-5)
        assert torch.allclose(
            text.encode(ids, am), clip.text_model(input_ids=ids, attention_mask=am).last_hidden_state, atol=1e-5
        )
        # embed() equals HF's projected features, normalized
        ref_img = F.normalize(clip.get_image_features(pixel_values=px).pooler_output, dim=-1)
        ref_txt = F.normalize(clip.get_text_features(input_ids=ids, attention_mask=am).pooler_output, dim=-1)
        assert torch.allclose(vision.embed(px), ref_img, atol=1e-5)
        assert torch.allclose(text.embed(ids, am), ref_txt, atol=1e-5)
        # masked paths with nothing masked equal the clean paths
        all_ids = torch.arange(vision.num_patches).expand(px.shape[0], -1)
        assert torch.allclose(vision.encode(px, all_ids), vision.encode(px), atol=1e-5)
        no_mask = torch.zeros_like(ids, dtype=torch.bool)
        assert torch.allclose(text.encode(ids, am, no_mask), text.encode(ids, am), atol=1e-6)


def test_tiny_towers_match_hf(tokenizer):
    check_equivalence(*towers_for(TINY_CLIP), tokenizer)


@pytest.mark.slow
def test_real_clip_towers_match_hf(tokenizer):
    check_equivalence(*towers_for(CLIP_NAME), tokenizer)


def test_vision_encode_keeps_cls_plus_visible_patches():
    _, vision, _ = towers_for(TINY_CLIP)
    ids_keep = torch.tensor([[0, 5, 9], [1, 2, 48]])
    out = vision.encode(pixels(2), ids_keep)
    assert out.shape == (2, 4, vision.hidden_size)


def test_text_mask_embedding_replaces_masked_tokens_and_gets_gradient(tokenizer):
    _, _, text = towers_for(TINY_CLIP)
    ids, am = text_inputs(tokenizer)
    token_mask = torch.zeros_like(ids, dtype=torch.bool)
    token_mask[:, 1] = True  # first word of every caption
    masked = text.encode(ids, am, token_mask)
    assert not torch.allclose(masked, text.encode(ids, am))
    masked.sum().backward()
    assert text.mask_embedding.grad is not None and text.mask_embedding.grad.abs().sum() > 0


def test_mean_pooling_embeds_are_normalized_and_use_new_projection(tokenizer):
    torch.manual_seed(0)
    clip = load_clip(TINY_CLIP)
    vision, text = ClipVisionTower(clip, pooling="mean"), ClipTextTower(clip, pooling="mean")
    ids, am = text_inputs(tokenizer)
    img, txt = vision.embed(pixels()), text.embed(ids, am)
    assert torch.allclose(img.norm(dim=-1), torch.ones(3)) and torch.allclose(txt.norm(dim=-1), torch.ones(3))
    assert vision.mean_projection is not None and text.mean_projection is not None


def test_parameter_groups():
    _, vision, text = towers_for(TINY_CLIP)
    pretrained_text = {id(p) for p in text.pretrained_parameters()}
    assert id(text.mask_embedding) not in pretrained_text
    assert {id(p) for p in text.native_head_parameters()} == {id(p) for p in text.projection.parameters()}
    head = {id(p) for p in vision.native_head_parameters()}
    assert {id(p) for p in vision.projection.parameters()} <= head
    assert {id(p) for p in vision.model.post_layernorm.parameters()} <= head


def test_build_backbone_registry():
    towers = build_backbone("hf_clip", TINY_CLIP)
    assert isinstance(towers.vision, ClipVisionTower) and isinstance(towers.text, ClipTextTower)
    assert towers.logit_scale.ndim == 0 and towers.logit_scale.requires_grad
    with pytest.raises(ValueError):
        build_backbone("open_clip", TINY_CLIP)
    with pytest.raises(ValueError):
        build_backbone("hf_clip", TINY_CLIP, pooling="max")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_backbones.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.models.backbones'`.

- [ ] **Step 4: Implement**

`mmae/models/backbones.py`:
```python
"""Image and text towers.

encode() returns hidden states for the decoders (optionally on masked inputs); embed() returns the
L2-normalized joint-space embedding of a clean input, used by the contrastive loss and retrieval.
The towers call HF CLIP's internal stages directly (embeddings, encoder, norms) so that masking can
happen on the input; tests/test_backbones.py pins them to HF's own outputs.
"""
from __future__ import annotations

from typing import Callable, Iterator, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPConfig, CLIPModel
from transformers.masking_utils import create_causal_mask

# A random tiny CLIP with CLIP's real vocabulary, special tokens and image size, so it works with the
# real tokenizer and preprocessing. Used by tests and CPU smoke runs.
TINY_CLIP = "tiny-random-clip"
POOLINGS = ("native", "mean")


def tiny_clip_config() -> CLIPConfig:
    return CLIPConfig(
        text_config=dict(
            vocab_size=49408,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=77,
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=49407,
        ),
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=32,
        ),
        projection_dim=24,
    )


def load_clip(pretrained: str) -> CLIPModel:
    if pretrained == TINY_CLIP:
        return CLIPModel(tiny_clip_config())
    return CLIPModel.from_pretrained(pretrained)


def _check_pooling(pooling: str) -> None:
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLINGS}")


class ClipVisionTower(nn.Module):
    """CLIP ViT with MAE-style token dropping: only CLS and the visible patches enter the transformer."""

    def __init__(self, clip: CLIPModel, pooling: str = "native") -> None:
        super().__init__()
        _check_pooling(pooling)
        self.model = clip.vision_model
        self.projection = clip.visual_projection
        config = self.model.config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.pooling = pooling
        self.mean_projection = (
            nn.Linear(self.hidden_size, clip.projection_dim, bias=False) if pooling == "mean" else None
        )

    def encode(self, pixel_values: torch.Tensor, ids_keep: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 1 + N_keep, hidden) hidden states of CLS and the visible patches (all if ids_keep is None)."""
        hidden = self.model.embeddings(pixel_values)  # (B, 1 + N, C), position embeddings already added
        if ids_keep is not None:
            index = ids_keep.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            hidden = torch.cat([hidden[:, :1], torch.gather(hidden[:, 1:], 1, index)], dim=1)
        hidden = self.model.pre_layrnorm(hidden)
        return self.model.encoder(inputs_embeds=hidden).last_hidden_state

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(pixel_values)
        if self.pooling == "native":
            z = self.projection(self.model.post_layernorm(tokens[:, 0]))
        else:
            z = self.mean_projection(tokens.mean(dim=1))
        return F.normalize(z, dim=-1)

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()
        yield from self.projection.parameters()

    def native_head_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters used only by native pooling (unused without the contrastive loss or with mean pooling)."""
        yield from self.model.post_layernorm.parameters()
        yield from self.projection.parameters()


class ClipTextTower(nn.Module):
    """CLIP text transformer with a learned [MASK] embedding replacing masked tokens on the input."""

    def __init__(self, clip: CLIPModel, pooling: str = "native") -> None:
        super().__init__()
        _check_pooling(pooling)
        self.model = clip.text_model
        self.projection = clip.text_projection
        config = self.model.config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_positions = config.max_position_embeddings
        # CLIP's tokenizer has no [MASK] token, so masked positions get this learned embedding.
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.mask_embedding, std=0.02)
        self.pooling = pooling
        self.mean_projection = (
            nn.Linear(self.hidden_size, clip.projection_dim, bias=False) if pooling == "mean" else None
        )

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """(B, T, hidden) final-layer-normed hidden states; token_mask (B, T) bool marks masked inputs."""
        tokens = self.model.embeddings.token_embedding(input_ids)
        if token_mask is not None:
            tokens = torch.where(token_mask.unsqueeze(-1), self.mask_embedding.to(tokens.dtype), tokens)
        hidden = self.model.embeddings(inputs_embeds=tokens)  # adds position embeddings
        causal = create_causal_mask(
            config=self.model.config, inputs_embeds=hidden, attention_mask=attention_mask, past_key_values=None
        )
        hidden = self.model.encoder(inputs_embeds=hidden, attention_mask=causal, is_causal=True).last_hidden_state
        return self.model.final_layer_norm(hidden)

    def eos_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Index of the pooled (EOS) token, located exactly as HF's CLIPTextModel does."""
        if self.model.eos_token_id == 2:  # legacy configs: EOS has the largest id
            return input_ids.to(torch.int).argmax(dim=-1)
        return (input_ids == self.model.eos_token_id).int().argmax(dim=-1)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encode(input_ids, attention_mask)
        if self.pooling == "native":
            rows = torch.arange(tokens.shape[0], device=tokens.device)
            z = self.projection(tokens[rows, self.eos_positions(input_ids).to(tokens.device)])
        else:
            weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
            z = self.mean_projection((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0))
        return F.normalize(z, dim=-1)

    def pretrained_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.model.parameters()
        yield from self.projection.parameters()

    def native_head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.projection.parameters()


class Towers(NamedTuple):
    vision: ClipVisionTower
    text: ClipTextTower
    logit_scale: nn.Parameter


def build_hf_clip(pretrained: str, pooling: str) -> Towers:
    clip = load_clip(pretrained)
    return Towers(ClipVisionTower(clip, pooling), ClipTextTower(clip, pooling), clip.logit_scale)


BACKBONES: dict[str, Callable[[str, str], Towers]] = {"hf_clip": build_hf_clip}


def build_backbone(kind: str, pretrained: str, pooling: str = "native") -> Towers:
    if kind not in BACKBONES:
        raise ValueError(f"unknown backbone type {kind!r}; choose from {sorted(BACKBONES)}")
    _check_pooling(pooling)
    return BACKBONES[kind](pretrained, pooling)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_backbones.py -q && $PY -m pytest -m slow tests/test_backbones.py -q && $PY -m pytest -q`
Expected: all pass (the slow test downloads nothing new: B/32 is in the HF cache).

- [ ] **Step 6: Confirm the guards bite, then commit**

Temporarily remove `hidden = self.model.pre_layrnorm(hidden)` from `ClipVisionTower.encode`: `test_tiny_towers_match_hf` must FAIL. Restore. Temporarily pass `attention_mask=None` to `create_causal_mask`: the text half of the same test must FAIL. Restore.

```bash
git add mmae/models/backbones.py tests/test_backbones.py tests/conftest.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add CLIP towers with input masking and HF-equivalence tests" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 4: Fusion modules and query decoders

**Files:**
- Create: `mmae/models/fusion.py`, `mmae/models/decoders.py`
- Test: `tests/test_fusion_decoders.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `mmae.models.fusion.FusionOutput` dataclass: `image_memory`, `image_padding`, `text_memory`, `text_padding` (tensors or None; padding is bool, True = padded).
  - `NoFusion()`, `ConcatFusion(dim, depth=0, heads=8)`, `MultiLearnerFusion(dim, heads=8, learner_depth=2, learner_ff_dim=512)`; each `forward(image: Tensor|None (B, Li, dim), text: Tensor|None (B, Lt, dim), text_padding: Tensor|None (B, Lt)) -> FusionOutput`.
  - `build_fusion(cfg) -> nn.Module` where `cfg` has `type`, `dim`, `depth`, `heads`, `learner_depth`, `learner_ff_dim`; registry `FUSIONS`.
  - `mmae.models.decoders.QueryDecoder(num_queries, dim, out_dim, depth=4, heads=8, dropout=0.1)`; `forward(memory (B, L, dim), memory_padding=None, query_padding=None (B, n)) -> (B, n, out_dim)` where `n = num_queries` or `query_padding.shape[1]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_fusion_decoders.py`:
```python
import pytest
import torch
from omegaconf import OmegaConf

from mmae.models.decoders import QueryDecoder
from mmae.models.fusion import ConcatFusion, MultiLearnerFusion, NoFusion, build_fusion

DIM = 16


def fusion_cfg(kind: str, depth: int = 0):
    return OmegaConf.create(dict(type=kind, dim=DIM, depth=depth, heads=4, learner_depth=2, learner_ff_dim=32))


def inputs(batch: int = 3):
    g = torch.Generator().manual_seed(0)
    image = torch.randn(batch, 5, DIM, generator=g)
    text = torch.randn(batch, 8, DIM, generator=g)
    lengths = torch.tensor([8, 5, 3])
    padding = torch.arange(8).unsqueeze(0) >= lengths.unsqueeze(1)
    return image, text, padding


@pytest.mark.parametrize("kind,depth", [("none", 0), ("concat", 0), ("concat", 2), ("multilearner", 0)])
def test_fusion_shapes(kind, depth):
    image, text, padding = inputs()
    out = build_fusion(fusion_cfg(kind, depth)).eval()(image, text, padding)
    if kind == "none":
        assert out.image_memory.shape == (3, 5, DIM) and out.image_padding is None
        assert out.text_memory.shape == (3, 8, DIM) and torch.equal(out.text_padding, padding)
    else:
        assert out.image_memory.shape == (3, 13, DIM) and out.text_memory.shape == (3, 13, DIM)
        assert torch.equal(out.image_padding[:, 5:], padding) and not out.image_padding[:, :5].any()


@pytest.mark.parametrize("kind", ["concat", "multilearner"])
def test_fusion_handles_one_modality(kind):
    image, text, padding = inputs()
    fusion = build_fusion(fusion_cfg(kind)).eval()
    assert fusion(image, None, None).image_memory.shape == (3, 5, DIM)
    assert fusion(None, text, padding).text_memory.shape == (3, 8, DIM)


def test_multilearner_gives_each_decoder_its_own_memory():
    image, text, padding = inputs()
    out = MultiLearnerFusion(DIM, heads=4, learner_depth=1, learner_ff_dim=32).eval()(image, text, padding)
    assert not torch.allclose(out.image_memory, out.text_memory)


def test_unknown_fusion_type():
    with pytest.raises(ValueError):
        build_fusion(fusion_cfg("cross_attention"))


@pytest.mark.parametrize("kind,depth", [("none", 0), ("concat", 0), ("concat", 2), ("multilearner", 0)])
def test_padded_text_positions_do_not_change_any_output(kind, depth):
    torch.manual_seed(0)
    fusion = build_fusion(fusion_cfg(kind, depth)).eval()
    image_decoder = QueryDecoder(7, DIM, 12, depth=2, heads=4).eval()
    text_decoder = QueryDecoder(8, DIM, 50, depth=2, heads=4).eval()
    image, text, padding = inputs()
    noisy = text.clone()
    noisy[padding] = torch.randn(int(padding.sum()), DIM) * 10  # garbage at padded positions only

    def run(text_tokens):
        out = fusion(image, text_tokens, padding)
        img = image_decoder(out.image_memory, out.image_padding)
        txt = text_decoder(out.text_memory, out.text_padding, query_padding=padding)
        return img, txt

    with torch.no_grad():
        img_a, txt_a = run(text)
        img_b, txt_b = run(noisy)
    assert torch.allclose(img_a, img_b, atol=1e-5)
    assert torch.allclose(txt_a[~padding], txt_b[~padding], atol=1e-5)


def test_query_decoder_shapes_and_query_count():
    decoder = QueryDecoder(10, DIM, 6, depth=1, heads=4)
    memory = torch.randn(2, 4, DIM)
    assert decoder(memory).shape == (2, 10, 6)
    assert decoder(memory, query_padding=torch.zeros(2, 7, dtype=torch.bool)).shape == (2, 7, 6)
    with pytest.raises(ValueError):
        decoder(memory, query_padding=torch.zeros(2, 11, dtype=torch.bool))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_fusion_decoders.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.models.decoders'`.

- [ ] **Step 3: Implement**

`mmae/models/decoders.py`:
```python
"""Query decoders: learned queries (one per output position) cross-attending to a memory sequence."""
from __future__ import annotations

import torch
from torch import nn


class QueryDecoder(nn.Module):
    """Predicts one output vector per query; works with a memory of any length."""

    def __init__(
        self, num_queries: int, dim: int, out_dim: int, depth: int = 4, heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        self.pos_embed = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerDecoderLayer(
            dim, heads, 4 * dim, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, depth, norm=nn.LayerNorm(dim))
        self.head = nn.Linear(dim, out_dim)

    @property
    def num_queries(self) -> int:
        return self.queries.shape[0]

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        query_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """memory (B, L, dim); paddings are bool, True = padded. Returns (B, n, out_dim)."""
        n = self.num_queries if query_padding is None else query_padding.shape[1]
        if n > self.num_queries:
            raise ValueError(f"{n} queries requested but the decoder has {self.num_queries}")
        target = (self.queries[:n] + self.pos_embed[:n]).unsqueeze(0).expand(memory.shape[0], -1, -1)
        out = self.decoder(
            target, memory, tgt_key_padding_mask=query_padding, memory_key_padding_mask=memory_padding
        )
        return self.head(out)
```

`mmae/models/fusion.py`:
```python
"""Fusion modules.

Each takes projected image tokens (B, Li, dim), projected text tokens (B, Lt, dim) and the text padding
(B, Lt, True = padded), any of them None for a single-modality model, and returns the memory each
decoder reads. Add a variant by writing one class with this interface and registering it in FUSIONS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class FusionOutput:
    image_memory: torch.Tensor | None
    image_padding: torch.Tensor | None
    text_memory: torch.Tensor | None
    text_padding: torch.Tensor | None


def concat_modalities(
    image: torch.Tensor | None,
    text: torch.Tensor | None,
    text_padding: torch.Tensor | None,
    type_embed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add modality-type embeddings and concatenate along the sequence; returns (sequence, padding)."""
    parts, paddings = [], []
    if image is not None:
        parts.append(image + type_embed[0])
        paddings.append(torch.zeros(image.shape[:2], dtype=torch.bool, device=image.device))
    if text is not None:
        parts.append(text + type_embed[1])
        if text_padding is None:
            text_padding = torch.zeros(text.shape[:2], dtype=torch.bool, device=text.device)
        paddings.append(text_padding)
    if not parts:
        raise ValueError("fusion needs at least one modality")
    return torch.cat(parts, dim=1), torch.cat(paddings, dim=1)


def _type_embedding(dim: int) -> nn.Parameter:
    embed = nn.Parameter(torch.empty(2, dim))
    nn.init.normal_(embed, std=0.02)
    return embed


class NoFusion(nn.Module):
    """Each decoder reads only its own modality (v0's parallel multimodal MAE)."""

    def forward(self, image, text, text_padding) -> FusionOutput:
        return FusionOutput(image, None, text, text_padding)


class ConcatFusion(nn.Module):
    """Concatenate both modalities, then optionally `depth` transformer layers; both decoders read it."""

    def __init__(self, dim: int, depth: int = 0, heads: int = 8) -> None:
        super().__init__()
        self.type_embed = _type_embedding(dim)
        self.encoder = None
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                dim, heads, 4 * dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)

    def forward(self, image, text, text_padding) -> FusionOutput:
        x, padding = concat_modalities(image, text, text_padding, self.type_embed)
        if self.encoder is not None:
            x = self.encoder(x, src_key_padding_mask=padding)
        return FusionOutput(x, padding, x, padding)


class _Learner(nn.Module):
    """Transformer encoder plus an output projection (v0's TransformerLearnerHead)."""

    def __init__(self, dim: int, heads: int, depth: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        return self.out(self.encoder(x, src_key_padding_mask=padding))


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class MultiLearnerFusion(nn.Module):
    """v0's multi-learner design: image, text and joint learners on the concatenated sequence.

    Image memory = MLP([image learner ; joint learner]); text memory = MLP([text learner ; joint learner]).
    """

    def __init__(self, dim: int, heads: int = 8, learner_depth: int = 2, learner_ff_dim: int = 512) -> None:
        super().__init__()
        self.type_embed = _type_embedding(dim)
        self.image_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.text_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.joint_learner = _Learner(dim, heads, learner_depth, learner_ff_dim)
        self.image_mlp = _mlp(2 * dim, learner_ff_dim, dim)
        self.text_mlp = _mlp(2 * dim, learner_ff_dim, dim)

    def forward(self, image, text, text_padding) -> FusionOutput:
        x, padding = concat_modalities(image, text, text_padding, self.type_embed)
        joint = self.joint_learner(x, padding)
        image_memory = self.image_mlp(torch.cat([self.image_learner(x, padding), joint], dim=-1))
        text_memory = self.text_mlp(torch.cat([self.text_learner(x, padding), joint], dim=-1))
        return FusionOutput(image_memory, padding, text_memory, padding)


FUSIONS: dict[str, Callable[[Any], nn.Module]] = {
    "none": lambda cfg: NoFusion(),
    "concat": lambda cfg: ConcatFusion(cfg.dim, depth=cfg.depth, heads=cfg.heads),
    "multilearner": lambda cfg: MultiLearnerFusion(
        cfg.dim, heads=cfg.heads, learner_depth=cfg.learner_depth, learner_ff_dim=cfg.learner_ff_dim
    ),
}


def build_fusion(cfg: Any) -> nn.Module:
    if cfg.type not in FUSIONS:
        raise ValueError(f"unknown fusion type {cfg.type!r}; choose from {sorted(FUSIONS)}")
    return FUSIONS[cfg.type](cfg)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_fusion_decoders.py -q && $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Confirm the guard bites, then commit**

Temporarily drop `memory_key_padding_mask=memory_padding` from `QueryDecoder.forward`: `test_padded_text_positions_do_not_change_any_output` must FAIL for the concat cases. Restore.

```bash
git add mmae/models/fusion.py mmae/models/decoders.py tests/test_fusion_decoders.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add fusion modules and query decoders" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 5: Losses

**Files:**
- Create: `mmae/losses.py`, `tests/_ddp_workers.py`
- Test: `tests/test_losses.py`
- Modify: `tests/helpers.py` (add `free_port`, `run_ddp_worker`)

**Interfaces:**
- Produces (in `mmae.losses`):
  - `contrastive_loss(image_emb: Tensor (B, D), text_emb: Tensor (B, D), logit_scale: Tensor (scalar, log scale), gather: bool = True) -> Tensor` scalar float32.
  - `patchify(images: Tensor (B, C, H, W), patch_size: int) -> Tensor (B, N, p*p*C)`, row-major patch order like the ViT.
  - `mae_loss(pred: Tensor (B, N, p*p*3), images: Tensor, mask: Tensor (B, N) bool, patch_size: int, norm_pix: bool = True) -> Tensor`.
  - `mlm_loss(logits: Tensor (B, T, V), input_ids: Tensor (B, T), token_mask: Tensor (B, T) bool) -> Tensor`.
  - `MAX_LOGIT_SCALE = 100.0`.
- Produces (tests): `helpers.free_port() -> int`, `helpers.run_ddp_worker(command: str, out_dir: Path, nproc: int = 2, *args: str) -> subprocess.CompletedProcess`.

- [ ] **Step 1: Add the DDP test helpers and worker**

Append to `tests/helpers.py`:
```python
import os
import socket
import subprocess
import sys


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
```

`tests/_ddp_workers.py`:
```python
"""Multi-process workers launched by tests through torchrun (CPU, gloo). Not collected by pytest."""
import argparse
from pathlib import Path

import torch
import torch.distributed as dist


def contrastive(out: Path) -> None:
    from mmae.losses import contrastive_loss

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    g = torch.Generator().manual_seed(0)
    image = torch.nn.functional.normalize(torch.randn(8, 6, generator=g), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(8, 6, generator=g), dim=-1)
    local = slice(rank * 4, (rank + 1) * 4)
    img = image[local].clone().requires_grad_(True)
    txt = text[local].clone().requires_grad_(True)
    loss = contrastive_loss(img, txt, torch.tensor(2.0), gather=True)
    loss.backward()
    torch.save({"loss": loss.detach(), "img_grad": img.grad, "txt_grad": txt.grad}, out / f"rank{rank}.pt")
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unprepared", action="store_true")
    args = parser.parse_args()
    {"contrastive": contrastive}[args.command](args.out)


if __name__ == "__main__":
    main()
```
(Task 8 adds a `retrieval` command to the same file.)

- [ ] **Step 2: Write the failing tests**

`tests/test_losses.py`:
```python
import pytest
import torch
import torch.nn.functional as F

from helpers import run_ddp_worker
from mmae.losses import MAX_LOGIT_SCALE, contrastive_loss, mae_loss, mlm_loss, patchify


def normalized(n: int, d: int, seed: int) -> torch.Tensor:
    return F.normalize(torch.randn(n, d, generator=torch.Generator().manual_seed(seed)), dim=-1)


def reference_infonce(image, text, scale):
    logits = scale * image @ text.T
    labels = torch.arange(image.shape[0])
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def test_contrastive_matches_reference():
    image, text = normalized(8, 6, 0), normalized(8, 6, 1)
    loss = contrastive_loss(image, text, torch.tensor(2.0))
    assert torch.allclose(loss, reference_infonce(image, text, torch.tensor(2.0).exp()), atol=1e-6)


def test_contrastive_clamps_logit_scale():
    image, text = normalized(8, 6, 0), normalized(8, 6, 1)
    loss = contrastive_loss(image, text, torch.tensor(10.0))  # exp(10) >> 100
    assert torch.allclose(loss, reference_infonce(image, text, torch.tensor(MAX_LOGIT_SCALE)), atol=1e-5)


def test_contrastive_gather_matches_single_process(tmp_path):
    result = run_ddp_worker("contrastive", tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [torch.load(tmp_path / f"rank{r}.pt") for r in range(2)]
    # same draws as the worker: one generator, image first then text
    g = torch.Generator().manual_seed(0)
    image = F.normalize(torch.randn(8, 6, generator=g), dim=-1).requires_grad_(True)
    text = F.normalize(torch.randn(8, 6, generator=g), dim=-1).requires_grad_(True)
    single = contrastive_loss(image, text, torch.tensor(2.0), gather=False)
    single.backward()
    # the mean of the per-rank losses is the single-process loss
    assert torch.allclose((ranks[0]["loss"] + ranks[1]["loss"]) / 2, single.detach(), atol=1e-6)
    # each rank's gradient (summed over ranks by the gather) is world_size x the single-process gradient
    for r in range(2):
        rows = slice(r * 4, (r + 1) * 4)
        assert torch.allclose(ranks[r]["img_grad"], 2 * image.grad[rows], atol=1e-6)
        assert torch.allclose(ranks[r]["txt_grad"], 2 * text.grad[rows], atol=1e-6)


def test_patchify_matches_vit_patch_order():
    images = torch.randn(2, 3, 64, 64)
    patches = patchify(images, 32)
    assert patches.shape == (2, 4, 32 * 32 * 3)
    for row in range(2):
        for col in range(2):
            expected = images[1, :, row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32].permute(1, 2, 0).reshape(-1)
            assert torch.equal(patches[1, row * 2 + col], expected)


@pytest.mark.parametrize("norm_pix", [True, False])
def test_mae_loss_counts_masked_patches_only(norm_pix):
    images = torch.randn(2, 3, 64, 64)
    mask = torch.tensor([[True, False, True, False], [False, False, False, True]])
    pred = torch.randn(2, 4, 32 * 32 * 3)
    loss = mae_loss(pred, images, mask, 32, norm_pix=norm_pix)
    changed = pred.clone()
    changed[~mask] += 5.0  # visible patches must not matter
    assert torch.allclose(mae_loss(changed, images, mask, 32, norm_pix=norm_pix), loss)
    target = patchify(images, 32)
    if norm_pix:
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
    manual = ((pred - target) ** 2).mean(-1)[mask].mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_mlm_loss_counts_masked_tokens_only():
    logits = torch.randn(2, 5, 11)
    ids = torch.randint(0, 11, (2, 5))
    mask = torch.tensor([[False, True, False, False, False], [False, False, True, True, False]])
    loss = mlm_loss(logits, ids, mask)
    assert torch.allclose(loss, F.cross_entropy(logits[mask], ids[mask]))
    changed = logits.clone()
    changed[~mask] += 3.0
    assert torch.allclose(mlm_loss(changed, ids, mask), loss)


def test_mlm_loss_with_no_masked_tokens_is_zero_with_grad():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    loss = mlm_loss(logits, torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.bool))
    assert loss.item() == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_losses_return_float32_for_bf16_inputs():
    image, text = normalized(4, 6, 0).bfloat16(), normalized(4, 6, 1).bfloat16()
    assert contrastive_loss(image, text, torch.tensor(2.0)).dtype == torch.float32
    pred = torch.randn(2, 4, 3072).bfloat16()
    assert mae_loss(pred, torch.randn(2, 3, 64, 64), torch.ones(2, 4, dtype=torch.bool), 32).dtype == torch.float32
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert mlm_loss(torch.randn(2, 5, 11).bfloat16(), torch.zeros(2, 5, dtype=torch.long), mask).dtype == torch.float32
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_losses.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.losses'`.

- [ ] **Step 4: Implement**

`mmae/losses.py`:
```python
"""Training losses: contrastive InfoNCE (optionally over the global batch), MAE and MLM."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

MAX_LOGIT_SCALE = 100.0  # CLIP clamps exp(logit_scale) at 100


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def contrastive_loss(
    image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor, gather: bool = True
) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalized embeddings, matched pairs on the diagonal.

    With `gather` and more than one process, each rank scores its local rows against the global batch
    (open_clip's local-loss formulation); gradients flow back through the gather. Every rank must have
    the same local batch size.
    """
    scale = logit_scale.float().exp().clamp(max=MAX_LOGIT_SCALE)
    image_emb, text_emb = image_emb.float(), text_emb.float()
    batch = image_emb.shape[0]
    if gather and _world_size() > 1:
        all_image = torch.cat(all_gather(image_emb), dim=0)
        all_text = torch.cat(all_gather(text_emb), dim=0)
        offset = dist.get_rank() * batch
    else:
        all_image, all_text, offset = image_emb, text_emb, 0
    labels = torch.arange(batch, device=image_emb.device) + offset
    logits_image = scale * image_emb @ all_text.T
    logits_text = scale * text_emb @ all_image.T
    return 0.5 * (F.cross_entropy(logits_image, labels) + F.cross_entropy(logits_text, labels))


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, p*p*C), patches in the ViT's row-major order, pixels as (p, p, C)."""
    b, c, h, w = images.shape
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch_size * patch_size * c)


def mae_loss(
    pred: torch.Tensor, images: torch.Tensor, mask: torch.Tensor, patch_size: int, norm_pix: bool = True
) -> torch.Tensor:
    """Mean over masked patches of the per-patch MSE between predicted and target pixels."""
    target = patchify(images.float(), patch_size)
    if norm_pix:  # normalize each target patch by its own mean and variance, as in the MAE paper
        target = (target - target.mean(dim=-1, keepdim=True)) / (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
    per_patch = ((pred.float() - target) ** 2).mean(dim=-1)
    weights = mask.to(per_patch.dtype)
    return (per_patch * weights).sum() / weights.sum().clamp(min=1.0)


def mlm_loss(logits: torch.Tensor, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at masked positions only; zero (with a gradient path) if nothing is masked."""
    if not token_mask.any():
        return logits.float().sum() * 0.0
    return F.cross_entropy(logits[token_mask].float(), input_ids[token_mask])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_losses.py -q && $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Confirm the guards bite, then commit**

Temporarily replace `offset = dist.get_rank() * batch` with `offset = 0`: `test_contrastive_gather_matches_single_process` must FAIL. Restore. Temporarily replace `torch.cat(all_gather(image_emb), dim=0)` with `torch.cat([t.detach() for t in all_gather(image_emb)], dim=0)`: the gradient assertion must FAIL. Restore.

```bash
git add mmae/losses.py tests/test_losses.py tests/_ddp_workers.py tests/helpers.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add contrastive, MAE and MLM losses" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 6: The MultiMAE model and its configs

**Files:**
- Create: `mmae/models/model.py`, `configs/config.yaml`, `configs/model/base.yaml`, `configs/model/fusion_concat.yaml`, `configs/model/fusion_multilearner.yaml`, `configs/model/fusion_none.yaml`, `configs/model/image_mae.yaml`, `configs/model/text_mlm.yaml`, `configs/data/coco.yaml`, `configs/train/default.yaml`
- Modify: `mmae/models/__init__.py`, `tests/helpers.py` (add `compose_cfg`, `make_batch`, `MODEL_NAMES`)
- Test: `tests/test_model.py`, `tests/test_configs.py`

**Interfaces:**
- Consumes: `random_patch_mask`, `random_token_mask` (Task 2); `build_backbone`, `ClipVisionTower.{encode,embed,num_patches,patch_size,hidden_size,pretrained_parameters,native_head_parameters}`, `ClipTextTower.{encode,embed,hidden_size,vocab_size,max_positions,...}` (Task 3); `build_fusion`, `QueryDecoder` (Task 4); `contrastive_loss`, `mae_loss`, `mlm_loss` (Task 5).
- Produces:
  - `mmae.models.MultiMAE(cfg: DictConfig  # cfg.model, max_text_len: int)`; attributes `use_image`, `use_text`, `has_contrastive`, `vision`, `text`, `logit_scale`; `forward(batch: dict) -> dict[str, Tensor]` with keys `loss` and `loss_contrastive` / `loss_mae` / `loss_mlm` as applicable; `embed_image(pixel_values) -> (B, D)`; `embed_text(input_ids, attention_mask) -> (B, D)`; `param_groups(lr, lr_backbone, weight_decay) -> list[dict]` (each group has `params`, `lr`, `weight_decay`, `name` in `{backbone,head}_{decay,no_decay}`).
  - Batch dict keys: `pixel_values` (B, 3, H, W) float, `input_ids` / `attention_mask` / `special_tokens_mask` (B, T) long.
  - Config tree (see the YAML below); `cfg.model.backbone.processor` names the tokenizer/image processor.
  - `helpers.compose_cfg(*overrides) -> DictConfig`, `helpers.make_batch(tokenizer, batch_size=4, max_len=32) -> dict`, `helpers.MODEL_NAMES`.

- [ ] **Step 1: Write the configs**

`configs/config.yaml`:
```yaml
# Entry config for train.py and evaluate.py. Group files (model/, data/, train/) override these values.
defaults:
  - _self_
  - model: fusion_concat
  - data: coco
  - train: default
  - override hydra/job_logging: none
  - override hydra/hydra_logging: none

seed: 42

paths:
  res_dir: res   # one folder per run: res/<wandb.project>/<wandb.group or default>/<time>_<name>/

wandb:
  enabled: true
  project: multimae
  entity: augustoxq
  group: ""
  name: ""       # run name; empty = the model config name
  tags: []
  notes: ""
  mode: online   # online | offline | disabled

eval:            # used by evaluate.py
  run_dir: null  # a res/ run folder; null = zero-shot pretrained backbone
  split: test
  output: null   # optional path: also write the metrics to this JSON file

hydra:           # Hydra writes nothing to disk; the run folder under res/ holds config and logs
  run:
    dir: .
  output_subdir: null
```

`configs/model/base.yaml`:
```yaml
# Shared model settings; every model/*.yaml starts from this file.
name: ???
modalities: [image, text]
backbone:
  type: hf_clip
  pretrained: openai/clip-vit-base-patch32   # or tiny-random-clip for tests and CPU smoke runs
  processor: openai/clip-vit-base-patch32    # tokenizer and image preprocessing
freeze_backbones: false
pooling: native          # native: CLIP's CLS/EOS pooling + pretrained projections; mean: mean pool + new projection
masking:
  image_ratio: 0.75
  text_ratio: 0.15
fusion:
  type: concat           # none | concat | multilearner
  dim: 256
  depth: 0               # concat: transformer layers after concatenation (0 = plain concat)
  heads: 8
  learner_depth: 2       # multilearner only
  learner_ff_dim: 512    # multilearner only
decoder:
  depth: 4
  heads: 8
  dropout: 0.1
loss:
  weights:
    contrastive: 1.0
    mae: 1.0
    mlm: 1.0
  gather: true           # contrastive negatives from the global batch under multi-GPU
  norm_pix: true         # MAE target: per-patch normalized pixels
monitor:                 # early stopping
  metric: val/retrieval/rsum
  mode: max
```

`configs/model/fusion_concat.yaml`:
```yaml
defaults:
  - base
  - _self_

name: fusion_concat
fusion:
  type: concat
```

`configs/model/fusion_multilearner.yaml`:
```yaml
defaults:
  - base
  - _self_

name: fusion_multilearner
fusion:
  type: multilearner
```

`configs/model/fusion_none.yaml`:
```yaml
# Parallel multimodal MAE: contrastive + MAE + MLM, each decoder reads only its own modality.
defaults:
  - base
  - _self_

name: fusion_none
fusion:
  type: none
```

`configs/model/image_mae.yaml`:
```yaml
# Image-only MAE on the CLIP vision tower (debugging and optional comparison).
defaults:
  - base
  - _self_

name: image_mae
modalities: [image]
fusion:
  type: none
monitor:
  metric: val/loss
  mode: min
```

`configs/model/text_mlm.yaml`:
```yaml
# Text-only MLM on the CLIP text tower (debugging and optional comparison).
defaults:
  - base
  - _self_

name: text_mlm
modalities: [text]
fusion:
  type: none
monitor:
  metric: val/loss
  mode: min
```

`configs/data/coco.yaml`:
```yaml
images_dir: /data/SSD/coco/images
annotations_dir: /data/SSD/coco/annotations   # coco_karpathy_{train,val,test}.json
max_text_len: 32
limit_train: null   # first N items of each split (null = all); for debug runs
limit_val: null
limit_test: null
```

`configs/train/default.yaml`:
```yaml
epochs: 10
batch_size: 128
eval_batch_size: 256
lr: 1.0e-4          # new modules; untested placeholder
lr_backbone: 1.0e-5 # CLIP towers; untested placeholder
weight_decay: 0.05
warmup_steps: 500
grad_clip: 1.0      # null disables
grad_accum: 1
precision: "no"     # "no" | bf16
eval_every: 1
patience: 5
min_delta: 1.0e-4
log_every: 50
num_workers: 8
save: best          # best | none
```

- [ ] **Step 2: Add test helpers**

Append to `tests/helpers.py`:
```python
import torch
import torch.nn.functional as F

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
```

- [ ] **Step 3: Write the failing tests**

`tests/test_configs.py`:
```python
import pytest

from helpers import MODEL_NAMES, compose_cfg


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_config_composes(name):
    cfg = compose_cfg(f"model={name}")
    assert cfg.model.name == name
    assert cfg.model.backbone.type == "hf_clip"
    both = set(cfg.model.modalities) == {"image", "text"}
    assert cfg.model.monitor.metric == ("val/retrieval/rsum" if both else "val/loss")
    assert cfg.train.lr == 1e-4 and cfg.train.lr_backbone == 1e-5
    assert cfg.wandb.project == "multimae" and cfg.paths.res_dir == "res"
```

`tests/test_model.py`:
```python
import pytest
import torch

from helpers import MODEL_NAMES, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP

EXPECTED_LOSSES = {
    "fusion_concat": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_multilearner": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "fusion_none": {"loss", "loss_contrastive", "loss_mae", "loss_mlm"},
    "image_mae": {"loss", "loss_mae"},
    "text_mlm": {"loss", "loss_mlm"},
}


def tiny_model(name: str, *overrides: str) -> MultiMAE:
    torch.manual_seed(0)
    cfg = compose_cfg(f"model={name}", f"model.backbone.pretrained={TINY_CLIP}", *overrides)
    return MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_returns_expected_finite_losses(name, tokenizer):
    out = tiny_model(name)(make_batch(tokenizer))
    assert set(out) == EXPECTED_LOSSES[name]
    assert all(torch.isfinite(v) and v.ndim == 0 for v in out.values())
    parts = sum(v for k, v in out.items() if k != "loss")
    assert torch.allclose(out["loss"], parts)  # all weights are 1.0 by default


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_trainable_parameter_gets_a_gradient(name, tokenizer):
    model = tiny_model(name)
    model(make_batch(tokenizer))["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing  # DDP (find_unused_parameters=False) needs this


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_frozen_backbones_get_no_gradient(name, tokenizer):
    model = tiny_model(name, "model.freeze_backbones=true")
    model(make_batch(tokenizer))["loss"].backward()
    for tower in (model.vision, model.text):
        if tower is not None:
            assert all(not p.requires_grad and p.grad is None for p in tower.pretrained_parameters())
    if model.text is not None:
        assert model.text.mask_embedding.grad is not None
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_mean_pooling_trains_the_new_projection(tokenizer):
    model = tiny_model("fusion_concat", "model.pooling=mean")
    model(make_batch(tokenizer))["loss"].backward()
    assert model.vision.mean_projection.weight.grad is not None
    assert not model.vision.projection.weight.requires_grad  # native head unused, frozen for DDP
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_tiny_batch_overfits(name, tokenizer):
    model = tiny_model(name)
    batch = make_batch(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(60):
        out = model(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        history.append({k: v.item() for k, v in out.items()})
    for key in ("loss_mae", "loss_mlm"):
        if key in history[0]:
            first = sum(h[key] for h in history[:5]) / 5
            last = sum(h[key] for h in history[-5:]) / 5
            assert last < 0.7 * first, (key, first, last)


def test_empty_caption_in_batch_is_finite(tokenizer):
    batch = make_batch(tokenizer, captions=["", "a dog", "", "two cats on a bed"])
    out = tiny_model("fusion_concat")(batch)
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()


def test_bf16_autocast_losses_are_finite_float32(tokenizer):
    model = tiny_model("fusion_multilearner")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(make_batch(tokenizer))
    assert all(v.dtype == torch.float32 and torch.isfinite(v) for v in out.values())


def test_embeddings_are_normalized(tokenizer):
    model = tiny_model("fusion_concat").eval()
    batch = make_batch(tokenizer)
    with torch.no_grad():
        img = model.embed_image(batch["pixel_values"])
        txt = model.embed_text(batch["input_ids"], batch["attention_mask"])
    assert torch.allclose(img.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(txt.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_param_groups_partition_trainable_parameters():
    model = tiny_model("fusion_concat")
    groups = model.param_groups(lr=1e-4, lr_backbone=1e-5, weight_decay=0.05)
    seen = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable) and len(seen) == len(set(seen))
    by_name = {g["name"]: g for g in groups}
    assert by_name["backbone_decay"]["lr"] == 1e-5 and by_name["head_decay"]["lr"] == 1e-4
    no_decay = {id(p) for g in groups if g["weight_decay"] == 0.0 for p in g["params"]}
    for param in (
        model.logit_scale,
        model.text.mask_embedding,
        model.image_decoder.queries,
        model.image_decoder.pos_embed,
        model.vision.model.embeddings.position_embedding.weight,
        model.text.model.embeddings.token_embedding.weight,
        model.fusion.type_embed,
    ):
        assert id(param) in no_decay
    assert id(model.image_proj.weight) not in no_decay


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        tiny_model("fusion_concat", "model.modalities=[audio]")
    with pytest.raises(ValueError):
        tiny_model("fusion_concat", "data.max_text_len=100")  # CLIP has 77 positions
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_model.py tests/test_configs.py -q`
Expected: config tests pass (configs exist); model tests FAIL with `ImportError: cannot import name 'MultiMAE' from 'mmae.models'`.

- [ ] **Step 5: Implement**

`mmae/models/model.py`:
```python
"""MultiMAE: CLIP towers with input masking, a fusion module and query decoders.

forward(batch) returns every loss, so the training loop and DDP see one ordinary forward pass:
  1. clean pass (both modalities only): contrastive loss on the towers' joint-space embeddings;
  2. masked pass: mask patches and tokens, encode the visible inputs, fuse, decode, and score
     MAE on masked patches and MLM on masked tokens.
"""
from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from mmae.losses import contrastive_loss, mae_loss, mlm_loss
from mmae.models.backbones import build_backbone
from mmae.models.decoders import QueryDecoder
from mmae.models.fusion import build_fusion
from mmae.models.masking import random_patch_mask, random_token_mask

# Parameter names (substrings) that get no weight decay, besides every parameter with ndim < 2.
NO_DECAY_KEYS = ("position_embedding", "token_embedding", "queries", "pos_embed", "type_embed")
MODALITIES = {"image", "text"}


class MultiMAE(nn.Module):
    def __init__(self, cfg: DictConfig, max_text_len: int) -> None:
        super().__init__()
        modalities = set(cfg.modalities)
        if not modalities or modalities - MODALITIES:
            raise ValueError(f"modalities must be a non-empty subset of {sorted(MODALITIES)}, got {cfg.modalities}")
        self.use_image = "image" in modalities
        self.use_text = "text" in modalities
        self.has_contrastive = self.use_image and self.use_text
        self.image_ratio = float(cfg.masking.image_ratio)
        self.text_ratio = float(cfg.masking.text_ratio)
        self.loss_weights = {k: float(v) for k, v in cfg.loss.weights.items()}
        self.gather = bool(cfg.loss.gather)
        self.norm_pix = bool(cfg.loss.norm_pix)

        towers = build_backbone(cfg.backbone.type, cfg.backbone.pretrained, cfg.pooling)
        dim, dec = int(cfg.fusion.dim), cfg.decoder
        self.vision = None
        self.text = None
        self.logit_scale = towers.logit_scale if self.has_contrastive else None
        if self.use_image:
            self.vision = towers.vision
            self.image_proj = nn.Linear(self.vision.hidden_size, dim)
            self.image_decoder = QueryDecoder(
                self.vision.num_patches, dim, self.vision.patch_size**2 * 3,
                depth=dec.depth, heads=dec.heads, dropout=dec.dropout,
            )
        if self.use_text:
            if max_text_len > towers.text.max_positions:
                raise ValueError(f"max_text_len {max_text_len} exceeds the text tower's {towers.text.max_positions}")
            self.text = towers.text
            self.text_proj = nn.Linear(self.text.hidden_size, dim)
            self.text_decoder = QueryDecoder(
                max_text_len, dim, self.text.vocab_size, depth=dec.depth, heads=dec.heads, dropout=dec.dropout
            )
        self.fusion = build_fusion(cfg.fusion)

        for tower in self.towers():
            # The native contrastive head is unused without the contrastive loss or under mean pooling;
            # freeze it so DDP does not wait for gradients that never arrive.
            if not self.has_contrastive or cfg.pooling == "mean":
                for param in tower.native_head_parameters():
                    param.requires_grad = False
            if cfg.freeze_backbones:
                for param in tower.pretrained_parameters():
                    param.requires_grad = False

    def towers(self) -> list[nn.Module]:
        return [t for t in (self.vision, self.text) if t is not None]

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision.embed(pixel_values)

    def embed_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text.embed(input_ids, attention_mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch.get("pixel_values")
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        losses: dict[str, torch.Tensor] = {}

        if self.has_contrastive:
            losses["contrastive"] = contrastive_loss(
                self.embed_image(images), self.embed_text(input_ids, attention_mask), self.logit_scale, self.gather
            )

        image_tokens = text_tokens = text_padding = None
        if self.use_image:
            ids_keep, patch_mask = random_patch_mask(
                images.shape[0], self.vision.num_patches, self.image_ratio, device=images.device
            )
            image_tokens = self.image_proj(self.vision.encode(images, ids_keep))
        if self.use_text:
            token_mask = random_token_mask(attention_mask, batch["special_tokens_mask"], self.text_ratio)
            text_tokens = self.text_proj(self.text.encode(input_ids, attention_mask, token_mask))
            text_padding = ~attention_mask.bool()

        fused = self.fusion(image_tokens, text_tokens, text_padding)
        if self.use_image:
            pred = self.image_decoder(fused.image_memory, fused.image_padding)
            losses["mae"] = mae_loss(pred, images, patch_mask, self.vision.patch_size, norm_pix=self.norm_pix)
        if self.use_text:
            logits = self.text_decoder(fused.text_memory, fused.text_padding, query_padding=text_padding)
            losses["mlm"] = mlm_loss(logits, input_ids, token_mask)

        out = {f"loss_{name}": value for name, value in losses.items()}
        out["loss"] = sum(self.loss_weights[name] * value for name, value in losses.items())
        return out

    def param_groups(self, lr: float, lr_backbone: float, weight_decay: float) -> list[dict]:
        """AdamW groups: pretrained tower weights at lr_backbone, everything else at lr; no decay on
        biases, norms, embeddings, queries and the logit scale."""
        backbone = {id(p) for tower in self.towers() for p in tower.pretrained_parameters()}
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim < 2 or any(key in name for key in NO_DECAY_KEYS)
            groups.setdefault((id(param) in backbone, no_decay), []).append(param)
        return [
            {
                "params": params,
                "lr": lr_backbone if is_backbone else lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "name": f"{'backbone' if is_backbone else 'head'}_{'no_decay' if no_decay else 'decay'}",
            }
            for (is_backbone, no_decay), params in sorted(groups.items())
        ]
```

`mmae/models/__init__.py`:
```python
"""Model components: towers, masking, fusion, decoders and the MultiMAE model."""
from mmae.models.model import MultiMAE

__all__ = ["MultiMAE"]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_model.py tests/test_configs.py -q && $PY -m pytest -q`
Expected: all pass. If `test_tiny_batch_overfits` fails, first confirm the gradient tests pass and that masks are resampled each step; only then raise the step count (never loosen the 0.7 threshold), and record the measured losses in the commit message.

- [ ] **Step 7: Confirm the guards bite, then commit**

Temporarily delete the `native_head_parameters` freezing loop: `test_every_trainable_parameter_gets_a_gradient[image_mae]` must FAIL. Restore. Temporarily delete the `freeze_backbones` loop: `test_frozen_backbones_get_no_gradient` must FAIL. Restore.

```bash
git add configs mmae/models tests/test_model.py tests/test_configs.py tests/helpers.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add the MultiMAE model and Hydra configs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 7: COCO data

**Files:**
- Create: `mmae/data/transforms.py`, `mmae/data/coco.py`, `mmae/data/collate.py`
- Modify: `mmae/data/__init__.py`, `tests/helpers.py` (add `make_fake_coco`), `tests/conftest.py` (add `fake_coco` fixture)
- Test: `tests/test_data.py`

**Interfaces:**
- Produces (exported from `mmae.data`):
  - `build_image_transform(processor_name: str) -> Callable[[PIL.Image], Tensor (3, crop, crop)]`.
  - `CocoPairs(images_dir, annotations_dir, split: "train"|"val"|"test", transform, limit: int | None = None)`; item `(Tensor, str)`.
  - `CocoRetrieval(images_dir, annotations_dir, split: "val"|"test", transform, limit=None)`; item `(Tensor, list[str] of 5)`.
  - `Collator(tokenizer_name: str, max_text_len: int)` with `.pairs(batch) -> {"pixel_values" (B,3,H,W), "input_ids", "attention_mask", "special_tokens_mask" (B,T)}` and `.retrieval(batch) -> {"pixel_values" (B,3,H,W), "input_ids", "attention_mask", "special_tokens_mask" (B,5,T)}`.
  - `CAPTIONS_PER_IMAGE = 5`.
  - Tests: `helpers.make_fake_coco(root: Path) -> tuple[Path, Path]` (images_dir, annotations_dir); fixture `fake_coco` returning that tuple.

- [ ] **Step 1: Add the fake COCO helper and fixture**

Append to `tests/helpers.py`:
```python
import json

from PIL import Image


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
```

Append to `tests/conftest.py`:
```python
from helpers import make_fake_coco


@pytest.fixture
def fake_coco(tmp_path):
    return make_fake_coco(tmp_path / "coco")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_data.py`:
```python
import pytest
import torch

from helpers import CLIP_NAME
from mmae.data import CAPTIONS_PER_IMAGE, Collator, CocoPairs, CocoRetrieval, build_image_transform

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@pytest.fixture(scope="module")
def transform():
    return build_image_transform(CLIP_NAME)


def test_transform_matches_clip_preprocessing(transform):
    from PIL import Image

    out = transform(Image.new("RGB", (320, 240), (255, 0, 128)))
    assert out.shape == (3, 224, 224)
    expected = [(c / 255 - m) / s for c, m, s in zip((255, 0, 128), CLIP_MEAN, CLIP_STD)]
    assert torch.allclose(out[:, 100, 100], torch.tensor(expected), atol=1e-2)


def test_train_pairs(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "train", transform)
    assert len(pairs) == 16
    image, caption = pairs[0]
    assert image.shape == (3, 224, 224) and caption == "train caption 0 0"


def test_odd_images_load_as_rgb(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "train", transform)
    for index in (2, 4, 6):  # grayscale, CMYK, truncated JPEG
        image, _ = pairs[index]
        assert image.shape == (3, 224, 224) and torch.isfinite(image).all()


def test_val_flattens_first_five_captions(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "val", transform)
    assert len(pairs) == 6 * CAPTIONS_PER_IMAGE
    assert [pairs.pairs[i][1] for i in range(5)] == [f"val 0 caption {k}" for k in range(5)]
    retrieval = CocoRetrieval(*fake_coco, "val", transform)
    assert len(retrieval) == 6
    image, captions = retrieval[0]
    assert image.shape == (3, 224, 224) and captions == [f"val 0 caption {k}" for k in range(5)]


def test_limit_takes_first_items_and_clamps(fake_coco, transform):
    assert len(CocoPairs(*fake_coco, "train", transform, limit=3)) == 3
    assert len(CocoPairs(*fake_coco, "train", transform, limit=1000)) == 16
    assert len(CocoRetrieval(*fake_coco, "test", transform, limit=2)) == 2


def test_bad_split(fake_coco, transform):
    with pytest.raises(ValueError):
        CocoPairs(*fake_coco, "dev", transform)
    with pytest.raises(ValueError):
        CocoRetrieval(*fake_coco, "train", transform)


def test_collator_pairs_and_truncation_keeps_eos(tokenizer):
    collate = Collator(CLIP_NAME, max_text_len=8)
    batch = collate.pairs([(torch.zeros(3, 224, 224), "a"), (torch.ones(3, 224, 224), "word " * 30)])
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["special_tokens_mask"].shape == (2, 8)
    assert batch["input_ids"][1, -1].item() == tokenizer.eos_token_id  # truncated caption still ends with EOS
    assert batch["attention_mask"][1].all()
    assert batch["special_tokens_mask"][0].tolist() == [1, 0, 1, 1, 1, 1, 1, 1]  # BOS a EOS PAD...


def test_collator_retrieval_shapes():
    collate = Collator(CLIP_NAME, max_text_len=8)
    batch = collate.retrieval([(torch.zeros(3, 224, 224), [f"c {k}" for k in range(5)])] * 3)
    assert batch["pixel_values"].shape == (3, 3, 224, 224)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (3, 5, 8)


@pytest.mark.slow
def test_real_coco_sizes(transform):
    root, ann = "/data/SSD/coco/images", "/data/SSD/coco/annotations"
    assert len(CocoPairs(root, ann, "train", transform)) == 566747
    assert len(CocoPairs(root, ann, "val", transform)) == 25000
    assert len(CocoRetrieval(root, ann, "test", transform)) == 5000
    image, captions = CocoRetrieval(root, ann, "test", transform)[0]
    assert image.shape == (3, 224, 224) and len(captions) == 5
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_data.py -q`
Expected: FAIL with `ImportError: cannot import name 'CAPTIONS_PER_IMAGE' from 'mmae.data'`.

- [ ] **Step 4: Implement**

`mmae/data/transforms.py`:
```python
"""Image preprocessing identical to OpenAI CLIP's: bicubic resize of the shortest side, center crop,
CLIP mean/std. Sizes and statistics come from the backbone's image processor config. (HF's own
image processor resizes slightly differently; the published zero-shot numbers use this pipeline.)"""
from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor


def build_image_transform(processor_name: str) -> T.Compose:
    processor = AutoImageProcessor.from_pretrained(processor_name)
    size = processor.size["shortest_edge"]
    crop = (processor.crop_size["height"], processor.crop_size["width"])
    return T.Compose(
        [
            T.Resize(size, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )
```

`mmae/data/coco.py`:
```python
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
```

`mmae/data/collate.py`:
```python
"""Batch collation: stack images and tokenize captions per batch with the backbone's tokenizer."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer


class Collator:
    def __init__(self, tokenizer_name: str, max_text_len: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = int(max_text_len)

    def tokenize(self, captions: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            captions,
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return {key: enc[key] for key in ("input_ids", "attention_mask", "special_tokens_mask")}

    def pairs(self, batch: list[tuple[torch.Tensor, str]]) -> dict[str, torch.Tensor]:
        images, captions = zip(*batch)
        return {"pixel_values": torch.stack(images), **self.tokenize(list(captions))}

    def retrieval(self, batch: list[tuple[torch.Tensor, list[str]]]) -> dict[str, torch.Tensor]:
        images, captions = zip(*batch)
        per_image = len(captions[0])
        flat = self.tokenize([c for group in captions for c in group])
        return {
            "pixel_values": torch.stack(images),
            **{key: value.view(len(images), per_image, -1) for key, value in flat.items()},
        }
```

`mmae/data/__init__.py`:
```python
"""COCO datasets, preprocessing and batch collation."""
from mmae.data.coco import CAPTIONS_PER_IMAGE, CocoPairs, CocoRetrieval
from mmae.data.collate import Collator
from mmae.data.transforms import build_image_transform

__all__ = ["CAPTIONS_PER_IMAGE", "CocoPairs", "CocoRetrieval", "Collator", "build_image_transform"]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_data.py -q && $PY -m pytest -m slow tests/test_data.py -q && $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Confirm the guard bites, then commit**

Temporarily remove `ImageFile.LOAD_TRUNCATED_IMAGES = True`: `test_odd_images_load_as_rgb` must FAIL. Restore.

```bash
git add mmae/data tests/test_data.py tests/helpers.py tests/conftest.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add COCO datasets, CLIP preprocessing and collation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 8: Retrieval evaluation

**Files:**
- Create: `mmae/engine/retrieval.py`, `tests/reference_v0_retrieval.py`
- Modify: `tests/_ddp_workers.py` (add `retrieval` command)
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: any model with `embed_image(pixel_values) -> (B, D)` and `embed_text(input_ids, attention_mask) -> (B, D)` (e.g. `MultiMAE`, Task 6); retrieval batches from `Collator.retrieval` (Task 7) with `input_ids`/`attention_mask` of shape (B, 5, T).
- Produces (in `mmae.engine.retrieval`):
  - `encode_retrieval_set(model, loader, accelerator: Accelerator | None = None) -> tuple[Tensor (N, D), Tensor (N, 5, D)]` (float32, dataset order, gathered across processes; raises `ValueError` for an unprepared loader under more than one process; restores the model's train/eval mode).
  - `retrieval_metrics(image_emb (N, D), caption_emb (N, K, D), chunk: int = 256) -> dict[str, float]` with keys `i2t_R1 i2t_R5 i2t_R10 t2i_R1 t2i_R5 t2i_R10 i2t_meanR i2t_medR i2t_mAP t2i_meanR t2i_medR t2i_mAP rsum` (recalls and mAP in percent).
  - `evaluate_retrieval(model, loader, accelerator=None) -> dict[str, float]`.

- [ ] **Step 1: Copy v0's metric code as the test reference**

`tests/reference_v0_retrieval.py` (v0's `calculate_metrics` and the metric half of `evalrank`, from `legacy-v0:src/hook/eval_fusionmmae.py`, made device-agnostic):
```python
"""v0's retrieval metrics (legacy-v0:src/hook/eval_fusionmmae.py), kept only as a test reference."""
import numpy as np
import torch


def calculate_metrics(inds, mappings, captions_per_image):
    num_queries = inds.size(0)
    AP_scores, all_ranks = [], []
    for query_idx in range(num_queries):
        correct_indices = mappings[query_idx].tolist()
        query_inds = inds[query_idx]
        if type(correct_indices) == int:
            correct_mask = query_inds == torch.tensor(correct_indices)
            ranks = correct_mask.nonzero(as_tuple=True)[-1].item() + 1
        else:
            ranks = []
            for correct_index in correct_indices:
                position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                ranks.append(position.item() + 1)
            assert len(ranks) == captions_per_image
        if type(ranks) != list:
            ranks = [ranks]
        all_ranks.extend(ranks)
        AP = 0
        for j, rank in enumerate(sorted(ranks), start=1):
            AP += j / rank
        AP /= captions_per_image
        AP_scores.append(AP)
    return np.mean(all_ranks), np.median(all_ranks), np.mean(AP_scores)


def v0_metrics(image_embeddings, text_embeddings, text_to_image_map, image_to_text_map):
    num_text, num_im = text_embeddings.shape[0], image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10, 50, 100]
    dist_matrix = text_embeddings @ image_embeddings.T
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    t2i = []
    for k in k_vals:
        correct = torch.eq(inds[:, :k], text_to_image_map.unsqueeze(-1)).any(dim=1)
        t2i.append(correct.sum().item() / num_text * 100)
    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(inds, text_to_image_map, 1)
    inds = torch.argsort(dist_matrix.T, dim=1, descending=True)
    i2t = []
    for k in k_vals:
        correct = torch.zeros((num_im,), dtype=torch.bool)
        for i in range(captions_per_image):
            correct = correct | torch.eq(inds[:, :k], image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
        i2t.append(correct.sum().item() / num_im * 100)
    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(inds, image_to_text_map, captions_per_image)
    return {
        "i2t_R1": round(i2t[0], 2), "i2t_R5": round(i2t[1], 2), "i2t_R10": round(i2t[2], 2),
        "i2t_meanR": int(round(meanR_i2t, 0)), "i2t_medR": int(round(medR_i2t, 0)),
        "i2t_mAP": round(mAP_i2t * 100, 2),
        "t2i_R1": round(t2i[0], 2), "t2i_R5": round(t2i[1], 2), "t2i_R10": round(t2i[2], 2),
        "t2i_meanR": int(round(meanR_t2i, 0)), "t2i_medR": int(round(medR_t2i, 0)),
        "t2i_mAP": round(mAP_t2i * 100, 2),
    }
```

- [ ] **Step 2: Add the multi-process retrieval worker**

Add to `tests/_ddp_workers.py` (and register it in `main`'s dict as `"retrieval": retrieval`; `retrieval` takes `(out, unprepared)`, so change the dispatch to `{"contrastive": lambda: contrastive(args.out), "retrieval": lambda: retrieval(args.out, args.unprepared)}[args.command]()`):
```python
class DummyEmbedder(torch.nn.Module):
    """Deterministic stand-in for MultiMAE's embed_image / embed_text."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.image = torch.nn.Linear(12, 8)
        self.token = torch.nn.Embedding(100, 8)

    def embed_image(self, pixel_values):
        return torch.nn.functional.normalize(self.image(pixel_values), dim=-1)

    def embed_text(self, input_ids, attention_mask):
        return torch.nn.functional.normalize(self.token(input_ids).mean(dim=1), dim=-1)


def retrieval_dataset(n: int = 43):
    g = torch.Generator().manual_seed(1)
    images = torch.randn(n, 12, generator=g)
    ids = torch.randint(0, 100, (n, 5, 6), generator=g)
    return [
        {"pixel_values": images[i], "input_ids": ids[i], "attention_mask": torch.ones(5, 6, dtype=torch.long)}
        for i in range(n)
    ]


def retrieval(out: Path, unprepared: bool) -> None:
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from mmae.engine.retrieval import evaluate_retrieval, encode_retrieval_set

    accelerator = Accelerator(cpu=True)
    loader = DataLoader(retrieval_dataset(), batch_size=8, collate_fn=torch.utils.data.default_collate)
    model = DummyEmbedder()
    if unprepared:
        model = accelerator.prepare(model)
        try:
            evaluate_retrieval(model, loader, accelerator)
        except ValueError:
            (out / f"raised{accelerator.process_index}").touch()
        return
    model, loader = accelerator.prepare(model, loader)
    images, captions = encode_retrieval_set(model, loader, accelerator)
    metrics = evaluate_retrieval(model, loader, accelerator)
    if accelerator.is_main_process:
        torch.save({"images": images, "captions": captions, "metrics": metrics}, out / "retrieval.pt")
```

- [ ] **Step 3: Write the failing tests**

`tests/test_retrieval.py`:
```python
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

import _ddp_workers
from helpers import run_ddp_worker
from mmae.engine.retrieval import encode_retrieval_set, evaluate_retrieval, retrieval_metrics
from reference_v0_retrieval import v0_metrics


def random_embeddings(n: int = 50, k: int = 5, d: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    captions = F.normalize(images.unsqueeze(1) + 0.9 * torch.randn(n, k, d, generator=g), dim=-1)
    return images, captions


def v0_on(images, captions):
    n, k, _ = captions.shape
    text_to_image = torch.arange(n).repeat_interleave(k)
    image_to_text = torch.arange(n * k).view(n, k)
    return v0_metrics(images, captions.reshape(n * k, -1), text_to_image, image_to_text)


def test_metrics_match_v0():
    images, captions = random_embeddings()
    new, old = retrieval_metrics(images, captions, chunk=7), v0_on(images, captions)
    for key, value in old.items():
        if key.endswith(("meanR", "medR")):
            assert int(round(new[key])) == value, key
        else:
            assert abs(new[key] - value) < 0.011, (key, new[key], value)
    assert abs(new["rsum"] - sum(new[f"{d}_R{k}"] for d in ("i2t", "t2i") for k in (1, 5, 10))) < 1e-9


def test_perfect_embeddings_give_full_recall():
    images, _ = random_embeddings(n=20)
    metrics = retrieval_metrics(images, images.unsqueeze(1).repeat(1, 5, 1))
    assert metrics["i2t_R1"] == 100.0 and metrics["t2i_R1"] == 100.0 and metrics["rsum"] == 600.0
    assert metrics["t2i_meanR"] == 1.0 and metrics["t2i_mAP"] == 100.0


def single_process_reference():
    model = _ddp_workers.DummyEmbedder()
    loader = DataLoader(_ddp_workers.retrieval_dataset(), batch_size=8, collate_fn=default_collate)
    model.train()
    images, captions = encode_retrieval_set(model, loader)
    assert model.training  # mode restored
    return images, captions, evaluate_retrieval(model, loader)


def test_two_processes_match_one(tmp_path):
    result = run_ddp_worker("retrieval", tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    two = torch.load(tmp_path / "retrieval.pt")
    images, captions, metrics = single_process_reference()
    assert two["images"].shape == (43, 8) and two["captions"].shape == (43, 5, 8)
    assert torch.allclose(two["images"], images, atol=1e-6) and torch.allclose(two["captions"], captions, atol=1e-6)
    # matmuls over differently sized batches may differ in the last bit, so compare approximately
    assert two["metrics"] == pytest.approx(metrics)


def test_unprepared_loader_raises_under_two_processes(tmp_path):
    result = run_ddp_worker("retrieval", tmp_path, 2, "--unprepared")
    assert result.returncode == 0, result.stderr[-3000:]
    assert (tmp_path / "raised0").exists() and (tmp_path / "raised1").exists()
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_retrieval.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.engine.retrieval'`.

- [ ] **Step 5: Implement**

`mmae/engine/retrieval.py`:
```python
"""COCO retrieval evaluation (image-to-text and text-to-image) on joint-space embeddings."""
from __future__ import annotations

import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderStateMixin

RECALL_KS = (1, 5, 10)


@torch.no_grad()
def encode_retrieval_set(
    model: torch.nn.Module, loader, accelerator: Accelerator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Image embeddings (N, D) and caption embeddings (N, K, D) in dataset order.

    With more than one process the loader must be prepared (sharded); embeddings are gathered batch
    by batch with captions kept as (B, K, D), so accelerate drops the samples it duplicated to even
    out the last batch.
    """
    if accelerator is not None and accelerator.num_processes > 1 and not isinstance(loader, DataLoaderStateMixin):
        raise ValueError(
            "encode_retrieval_set needs a loader prepared by accelerate when running on "
            f"{accelerator.num_processes} processes; pass accelerator.prepare(loader)"
        )
    net = accelerator.unwrap_model(model) if accelerator is not None else model
    was_training = net.training
    net.eval()
    images, captions = [], []
    for batch in loader:
        b, k, t = batch["input_ids"].shape
        image = net.embed_image(batch["pixel_values"])
        caption = net.embed_text(
            batch["input_ids"].reshape(b * k, t), batch["attention_mask"].reshape(b * k, t)
        ).view(b, k, -1)
        if accelerator is not None:
            image = accelerator.gather_for_metrics(image)
            caption = accelerator.gather_for_metrics(caption)
        images.append(image.float())
        captions.append(caption.float())
    net.train(was_training)
    return torch.cat(images), torch.cat(captions)


def _ranks(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """1-based rank of each positive: 1 + number of candidates scored strictly higher.

    scores (Q, M), positives (Q, P) candidate indices -> (Q, P).
    """
    positive_scores = scores.gather(1, positives)
    return 1 + (scores.unsqueeze(1) > positive_scores.unsqueeze(-1)).sum(dim=-1)


def retrieval_metrics(image_emb: torch.Tensor, caption_emb: torch.Tensor, chunk: int = 256) -> dict[str, float]:
    """Recall@{1,5,10}, mean/median rank, mAP for i2t and t2i, and rsum (sum of the six recalls).

    i2t: an image query is correct at k if any of its K captions ranks <= k; its mean/median rank and
    mAP use all K captions. t2i: each caption has one positive image. Recalls and mAP are in percent.
    """
    n, k, d = caption_emb.shape
    device = image_emb.device
    text_emb = caption_emb.reshape(n * k, d)
    caption_ids = torch.arange(n * k, device=device).view(n, k)
    i2t, t2i = [], []
    for start in range(0, n, chunk):
        scores = image_emb[start : start + chunk] @ text_emb.T
        i2t.append(_ranks(scores, caption_ids[start : start + chunk]))
    for start in range(0, n * k, chunk):
        stop = min(start + chunk, n * k)
        scores = text_emb[start:stop] @ image_emb.T
        image_ids = (torch.arange(start, stop, device=device) // k).unsqueeze(1)
        t2i.append(_ranks(scores, image_ids).squeeze(1))
    i2t_ranks, t2i_ranks = torch.cat(i2t).double(), torch.cat(t2i).double()

    best = i2t_ranks.min(dim=1).values
    sorted_ranks = i2t_ranks.sort(dim=1).values
    average_precision = (torch.arange(1, k + 1, device=device, dtype=torch.double) / sorted_ranks).mean(dim=1)
    metrics: dict[str, float] = {}
    for top in RECALL_KS:
        metrics[f"i2t_R{top}"] = 100.0 * (best <= top).double().mean().item()
        metrics[f"t2i_R{top}"] = 100.0 * (t2i_ranks <= top).double().mean().item()
    metrics.update(
        i2t_meanR=i2t_ranks.mean().item(),
        i2t_medR=torch.quantile(i2t_ranks.flatten(), 0.5).item(),
        i2t_mAP=100.0 * average_precision.mean().item(),
        t2i_meanR=t2i_ranks.mean().item(),
        t2i_medR=torch.quantile(t2i_ranks, 0.5).item(),
        t2i_mAP=100.0 * (1.0 / t2i_ranks).mean().item(),
    )
    metrics["rsum"] = sum(metrics[f"{direction}_R{top}"] for direction in ("i2t", "t2i") for top in RECALL_KS)
    return metrics


def evaluate_retrieval(model: torch.nn.Module, loader, accelerator: Accelerator | None = None) -> dict[str, float]:
    images, captions = encode_retrieval_set(model, loader, accelerator)
    return retrieval_metrics(images, captions)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_retrieval.py -q && $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Confirm the guards bite, then commit**

Temporarily gather `caption` after flattening it to `(B*K, D)` (move `.view(b, k, -1)` after the gather): `test_two_processes_match_one` must FAIL. Restore. Temporarily change `_ranks` to `(scores.unsqueeze(1) >= ...)`: `test_perfect_embeddings_give_full_recall` must FAIL. Restore.

```bash
git add mmae/engine/retrieval.py tests/test_retrieval.py tests/reference_v0_retrieval.py tests/_ddp_workers.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add retrieval evaluation with per-batch gather" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 9: Run folders, logging and `list_runs.py`

**Files:**
- Create: `mmae/utils/run.py`, `mmae/utils/logging.py`, `scripts/list_runs.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: the composed config (Task 6) for `cfg.paths.res_dir`, `cfg.wandb.{project,group,name,tags,enabled,...}`, `cfg.model.name`.
- Produces:
  - `mmae.utils.run.Run(cfg: DictConfig, enabled: bool = True, now: datetime | None = None)`, a context manager. Attributes `name`, `path` (`Path | None`; None when disabled). Methods `log_metrics(metrics: dict, step: int) -> None`, `update(**fields) -> None` (merged into `run.json`), `save_checkpoint(name: str, obj) -> Path | None`. `__exit__` never swallows exceptions.
  - `mmae.utils.run.update_run_json(run_dir: Path, **fields) -> None`.
  - `mmae.utils.logging.setup_logging(is_main: bool) -> None`; `MetricLogger(run: Run, wandb_run=None)` with `log(metrics: dict, step: int) -> None`; `init_wandb(cfg, run: Run)` returning a wandb run or None.

- [ ] **Step 1: Write the failing tests**

`tests/test_run.py`:
```python
import json
import logging
import subprocess
import sys
from datetime import datetime

import pytest
from omegaconf import OmegaConf

from helpers import REPO, compose_cfg
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run, update_run_json

NOW = datetime(2026, 10, 1, 12, 0, 0)


def cfg_for(tmp_path, *overrides):
    return compose_cfg(f"paths.res_dir={tmp_path / 'res'}", "wandb.enabled=false", *overrides)


def test_run_folder_layout_and_lifecycle(tmp_path):
    cfg = cfg_for(tmp_path, "wandb.group=ablation")
    with Run(cfg, now=NOW) as run:
        assert run.path == tmp_path / "res" / "multimae" / "ablation" / "20261001_120000_fusion_concat"
        assert json.loads((run.path / "run.json").read_text())["status"] == "running"
        logging.getLogger("mmae.test").info("hello from the run")
        MetricLogger(run).log({"train/loss": 2.0, "epoch": 1}, step=10)
        MetricLogger(run).log({"val/loss": 1.5, "epoch": 1}, step=20)
        assert not (run.path / "checkpoints").exists()
        ckpt = run.save_checkpoint("best.pt", {"x": 1})
        assert ckpt == run.path / "checkpoints" / "best.pt" and ckpt.exists()
        run.update(results={"best_epoch": 1})
    info = json.loads((run.path / "run.json").read_text())
    assert info["status"] == "completed" and info["results"] == {"best_epoch": 1}
    assert {"commit", "dirty"} <= set(info["git"]) and info["duration_s"] >= 0
    assert OmegaConf.load(run.path / "config.yaml") == OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    lines = [json.loads(line) for line in (run.path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0] == {"step": 10, "metrics": {"train/loss": 2.0, "epoch": 1}}
    assert "hello from the run" in (run.path / "train.log").read_text()
    assert (run.path / "plots" / "curves.png").exists()
    assert not (run.path / "error.txt").exists()


def test_disabled_run_writes_nothing(tmp_path):
    with Run(cfg_for(tmp_path), enabled=False) as run:
        run.log_metrics({"a": 1.0}, step=1)
        run.update(x=1)
        assert run.save_checkpoint("best.pt", {}) is None and run.path is None
    assert not (tmp_path / "res").exists()


def test_same_second_same_name_gets_distinct_folders(tmp_path):
    cfg = cfg_for(tmp_path, "wandb.name=seed")
    with Run(cfg, now=NOW) as first, Run(cfg, now=NOW) as second:
        assert first.path != second.path
        assert second.path.name == "20261001_120000_seed_1"


def test_crash_marks_run_failed_and_reraises(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with Run(cfg_for(tmp_path), now=NOW) as run:
            raise RuntimeError("boom")
    info = json.loads((run.path / "run.json").read_text())
    assert info["status"] == "failed"
    assert "RuntimeError: boom" in (run.path / "error.txt").read_text()


def test_update_run_json_merges(tmp_path):
    with Run(cfg_for(tmp_path), now=NOW) as run:
        pass
    update_run_json(run.path, eval={"test": {"rsum": 1.0}})
    info = json.loads((run.path / "run.json").read_text())
    assert info["eval"] == {"test": {"rsum": 1.0}} and info["status"] == "completed"


def test_list_runs_filters(tmp_path):
    with Run(cfg_for(tmp_path, "wandb.group=a"), now=NOW):
        pass
    with pytest.raises(RuntimeError):
        with Run(cfg_for(tmp_path, "wandb.group=b"), now=NOW):
            raise RuntimeError("x")
    script = [sys.executable, str(REPO / "scripts" / "list_runs.py"), "--root", str(tmp_path / "res")]
    everything = subprocess.run(script, capture_output=True, text=True, check=True).stdout
    assert "completed" in everything and "failed" in everything
    failed = subprocess.run(script + ["--status", "failed"], capture_output=True, text=True, check=True).stdout
    assert "failed" in failed and "completed" not in failed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.utils.logging'`.

- [ ] **Step 3: Implement**

`mmae/utils/run.py`:
```python
"""One results folder per training run, adapted from CoSiR's ExperimentManager.

    res/<wandb.project>/<wandb.group or default>/<YYYYMMDD_HHMMSS>_<name>[_n]/
        config.yaml    resolved Hydra config
        run.json       status, timings, git commit, command, host, wandb, results
        metrics.jsonl  one line per logged step
        train.log      main-process console log
        error.txt      traceback, only if the run failed
        checkpoints/   created on first save
        plots/         learning curves drawn at the end

Only the main process writes; other ranks get a disabled Run whose methods do nothing. There is no
shared registry file (parallel runs would race on it); scripts/list_runs.py scans run.json files.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
PLOTTED_PREFIXES = ("train/loss", "val/loss", "val/retrieval/rsum", "val/retrieval/i2t_R1", "val/retrieval/t2i_R1")


def git_info(cwd: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def update_run_json(run_dir: Path, **fields: Any) -> None:
    path = Path(run_dir) / "run.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    info.update(fields)
    _write_json(path, info)


class Run:
    def __init__(self, cfg: DictConfig, enabled: bool = True, now: datetime | None = None) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.name = cfg.wandb.name or cfg.model.name
        self.path: Path | None = None
        self._now = now
        self._info: dict[str, Any] = {}
        self._start = 0.0
        self._handler: logging.Handler | None = None

    def _make_dir(self) -> Path:
        stamp = (self._now or datetime.now()).strftime("%Y%m%d_%H%M%S")
        base = Path(self.cfg.paths.res_dir) / self.cfg.wandb.project / (self.cfg.wandb.group or "default")
        base.mkdir(parents=True, exist_ok=True)
        for attempt in range(1000):
            path = base / (f"{stamp}_{self.name}" + (f"_{attempt}" if attempt else ""))
            try:
                path.mkdir()  # atomic: two runs in the same second cannot both get this name
                return path
            except FileExistsError:
                continue
        raise RuntimeError(f"no free run folder name under {base}")

    def __enter__(self) -> "Run":
        if not self.enabled:
            return self
        self.path = self._make_dir()
        OmegaConf.save(self.cfg, self.path / "config.yaml", resolve=True)
        self._start = time.time()
        self._info = {
            "name": self.name,
            "status": "running",
            "model": self.cfg.model.name,
            "group": self.cfg.wandb.group or "default",
            "tags": list(self.cfg.wandb.tags),
            "path": str(self.path.resolve()),
            "created": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "host": socket.gethostname(),
            "num_processes": int(os.environ.get("WORLD_SIZE", "1")),
            "git": git_info(Path.cwd()),
        }
        _write_json(self.path / "run.json", self._info)
        self._handler = logging.FileHandler(self.path / "train.log")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(self._handler)
        if root.level > logging.INFO or root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        self._info.update(
            status="failed" if exc_type else "completed",
            ended=datetime.now().isoformat(timespec="seconds"),
            duration_s=round(time.time() - self._start, 1),
        )
        if exc_type is not None:
            (self.path / "error.txt").write_text("".join(traceback.format_exception(exc_type, exc, tb)))
        _write_json(self.path / "run.json", self._info)
        try:
            self._plot()
        except Exception:  # plotting must never hide the run's own result
            log.exception("could not draw learning curves")
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()
        return False  # never swallow the exception

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled:
            return
        with open(self.path / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "metrics": metrics}, default=float) + "\n")

    def update(self, **fields: Any) -> None:
        if not self.enabled:
            return
        self._info.update(fields)
        _write_json(self.path / "run.json", self._info)

    def save_checkpoint(self, name: str, obj: Any) -> Path | None:
        if not self.enabled:
            return None
        directory = self.path / "checkpoints"
        directory.mkdir(exist_ok=True)
        torch.save(obj, directory / name)
        return directory / name

    def _plot(self) -> None:
        metrics_file = self.path / "metrics.jsonl"
        if not metrics_file.exists():
            return
        series: dict[str, list[tuple[int, float]]] = {}
        for line in metrics_file.read_text().splitlines():
            entry = json.loads(line)
            for key, value in entry["metrics"].items():
                if key.startswith(PLOTTED_PREFIXES) and isinstance(value, (int, float)):
                    series.setdefault(key, []).append((entry["step"], value))
        if not series:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = min(3, len(series))
        rows = -(-len(series) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
        for ax, (key, points) in zip(axes.flat, sorted(series.items())):
            steps, values = zip(*points)
            ax.plot(steps, values, marker="o" if len(points) < 30 else None)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.3)
        for ax in list(axes.flat)[len(series):]:
            ax.axis("off")
        fig.tight_layout()
        (self.path / "plots").mkdir(exist_ok=True)
        fig.savefig(self.path / "plots" / "curves.png", dpi=120)
        plt.close(fig)
```

`mmae/utils/logging.py`:
```python
"""Console logging, metric logging (run folder + wandb) and wandb setup."""
from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf

from mmae.utils.run import Run

log = logging.getLogger(__name__)
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "huggingface_hub", "PIL", "matplotlib")


def setup_logging(is_main: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class MetricLogger:
    """Writes metrics to the run folder and wandb; prints val/test metrics to the console."""

    def __init__(self, run: Run, wandb_run: Any = None) -> None:
        self.run = run
        self.wandb_run = wandb_run

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.run.log_metrics(metrics, step)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        if any(key.startswith(("val/", "test/")) for key in metrics):
            shown = ", ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float)))
            log.info("step %d: %s", step, shown)


def init_wandb(cfg: DictConfig, run: Run):
    """Start a wandb run (files under the run folder) and record its id/url in run.json; None if disabled."""
    if not cfg.wandb.enabled:
        return None
    import wandb

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        group=cfg.wandb.group or None,
        name=run.name,
        tags=list(cfg.wandb.tags) or None,
        notes=cfg.wandb.notes or None,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run.path) if run.path else None,
    )
    run.update(wandb={"id": wandb_run.id, "url": wandb_run.url})
    return wandb_run
```

`scripts/list_runs.py`:
```python
"""List training runs under res/ by scanning their run.json files.

Usage:
  python scripts/list_runs.py [--root res] [--group G] [--status completed|failed|running] [--model M] [--tag T]
"""
import argparse
import json
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("**/run.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("res"))
    parser.add_argument("--group")
    parser.add_argument("--status")
    parser.add_argument("--model")
    parser.add_argument("--tag")
    args = parser.parse_args()
    rows = []
    for run in load_runs(args.root):
        if args.group and run.get("group") != args.group:
            continue
        if args.status and run.get("status") != args.status:
            continue
        if args.model and run.get("model") != args.model:
            continue
        if args.tag and args.tag not in run.get("tags", []):
            continue
        results = run.get("results") or {}
        val = (results.get("best_val") or {}).get("val/retrieval/rsum")
        test = (results.get("test") or {}).get("test/retrieval/rsum")
        rows.append(
            [
                run.get("created", ""), run.get("status", ""), run.get("group", ""), run.get("name", ""),
                str(results.get("best_epoch", "")), "" if val is None else f"{val:.2f}",
                "" if test is None else f"{test:.2f}", run.get("path", ""),
            ]
        )
    header = ["created", "status", "group", "name", "best_ep", "val_rsum", "test_rsum", "path"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    for line in [header, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_run.py -q && $PY -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Confirm the guards bite, then commit**

Temporarily change `path.mkdir()` to `path.mkdir(exist_ok=True)`: `test_same_second_same_name_gets_distinct_folders` must FAIL. Restore. Temporarily make `__exit__` return `True`: `test_crash_marks_run_failed_and_reraises` must FAIL. Restore.

```bash
git add mmae/utils tests/test_run.py scripts/list_runs.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add res/ run folders, metric logging and list_runs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 10: Trainer and `train.py`

**Files:**
- Create: `mmae/engine/trainer.py`, `train.py`, `configs/train/debug.yaml`, `configs/data/coco_cluster.yaml`
- Modify: `mmae/engine/__init__.py`, `tests/helpers.py` (add `run_train`)
- Test: `tests/test_trainer.py`, `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `MultiMAE` (Task 6), `CocoPairs`, `CocoRetrieval`, `Collator`, `build_image_transform` (Task 7), `evaluate_retrieval` (Task 8), `Run`, `MetricLogger`, `setup_logging`, `init_wandb` (Task 9).
- Produces:
  - `mmae.engine.trainer.warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float`.
  - `mmae.engine.trainer.EarlyStopper(mode: "max"|"min", patience: int, min_delta: float)`: `update(value) -> bool` (True = new best), `should_stop` property, `best`.
  - `mmae.engine.trainer.Trainer(cfg, accelerator, run, metric_logger)`: `fit() -> dict` (`{"best_epoch": int|None, "best_val": dict, "test": dict}`, also written to `run.json` as `results`); `evaluate(split: "val"|"test") -> dict[str, float]` with keys `<split>/loss`, `<split>/loss_*` and, for two-modality models, `<split>/retrieval/*`; attributes `model`, `accelerator`, `global_step`.
  - `helpers.run_train(cwd: Path, fake_coco, *overrides, nproc: int = 1) -> subprocess.CompletedProcess`.

- [ ] **Step 1: Write the debug and cluster configs**

`configs/train/debug.yaml`:
```yaml
# @package _global_
# Tiny, fast run: 1 epoch on a few hundred COCO items, wandb off, no checkpoint.
# This file replaces train/default.yaml, so it repeats every train key; keep them in sync.
train:
  epochs: 1
  batch_size: 8
  eval_batch_size: 16
  lr: 1.0e-4
  lr_backbone: 1.0e-5
  weight_decay: 0.05
  warmup_steps: 2
  grad_clip: 1.0
  grad_accum: 1
  precision: "no"
  eval_every: 1
  patience: 5
  min_delta: 1.0e-4
  log_every: 2
  num_workers: 2
  save: none
data:
  limit_train: 256
  limit_val: 80
  limit_test: 80
wandb:
  enabled: false
```

`configs/data/coco_cluster.yaml`:
```yaml
# COCO on the cluster nodes' local disk.
defaults:
  - coco
  - _self_

images_dir: /local/wding/Dataset/coco/images
annotations_dir: /local/wding/Dataset/coco/annotations
```

- [ ] **Step 2: Add the smoke-run helper**

Append to `tests/helpers.py`:
```python
def run_train(cwd: Path, fake_coco, *overrides: str, nproc: int = 1, script: str = "train.py") -> subprocess.CompletedProcess:
    """Run train.py (or evaluate.py) on the fake COCO with the tiny CLIP on CPU, from `cwd`."""
    images_dir, annotations_dir = fake_coco
    args = [
        f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
        "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
        f"paths.res_dir={cwd / 'res'}", *overrides,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "1", "WANDB_MODE": "disabled"}
    if nproc == 1:
        cmd = [sys.executable, str(REPO / script), *args]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
               "--master_port", str(free_port()), str(REPO / script), *args]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1200)
```

- [ ] **Step 3: Write the failing tests**

`tests/test_trainer.py`:
```python
import hashlib
import math

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from helpers import compose_cfg
from mmae.engine.trainer import EarlyStopper, Trainer, warmup_cosine
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run


@pytest.fixture
def accelerator():
    acc = Accelerator(cpu=True)
    yield acc
    AcceleratorState._reset_state(True)


def tiny_cfg(tmp_path, fake_coco, *overrides):
    images_dir, annotations_dir = fake_coco
    return compose_cfg(
        f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
        "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
        f"paths.res_dir={tmp_path / 'res'}", *overrides,
    )


def weights_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_warmup_cosine():
    assert warmup_cosine(0, 10, 110) == pytest.approx(0.1)
    assert warmup_cosine(9, 10, 110) == pytest.approx(1.0)
    assert warmup_cosine(10, 10, 110) == pytest.approx(1.0)
    assert warmup_cosine(60, 10, 110) == pytest.approx(0.5)
    assert warmup_cosine(110, 10, 110) == pytest.approx(0.0, abs=1e-12)
    assert warmup_cosine(0, 0, 100) == pytest.approx(1.0)


def test_early_stopper():
    stop = EarlyStopper("max", patience=2, min_delta=0.5)
    assert stop.update(10.0) and stop.best == 10.0
    assert not stop.update(10.4) and not stop.should_stop  # within min_delta
    assert stop.update(11.0)
    assert not stop.update(9.0) and not stop.update(9.0) and stop.should_stop
    low = EarlyStopper("min", patience=1, min_delta=0.0)
    assert low.update(3.0) and low.update(2.0) and not low.update(2.5) and low.should_stop
    with pytest.raises(ValueError):
        EarlyStopper("best", 1, 0.0)


def test_fit_restores_best_weights_before_test(tmp_path, fake_coco, accelerator, monkeypatch):
    cfg = tiny_cfg(tmp_path, fake_coco, "train.epochs=3")
    scripted = iter([10.0, 30.0, 20.0])  # best at epoch 2
    hashes: dict[str, str] = {}

    def fake_evaluate(self, split):
        current = weights_hash(self.accelerator.unwrap_model(self.model))
        if split == "val":
            hashes[f"val{sum(k.startswith('val') for k in hashes) + 1}"] = current
            return {"val/loss": 1.0, "val/retrieval/rsum": next(scripted)}
        hashes["test"] = current
        return {"test/loss": 1.0, "test/retrieval/rsum": 0.0}

    monkeypatch.setattr(Trainer, "evaluate", fake_evaluate)
    with Run(cfg) as run:
        results = Trainer(cfg, accelerator, run, MetricLogger(run)).fit()
    assert results["best_epoch"] == 2
    assert hashes["test"] == hashes["val2"] != hashes["val3"]


def test_evaluate_reports_losses_and_retrieval(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco)
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run))
        first, second = trainer.evaluate("val"), trainer.evaluate("val")
    assert {"val/loss", "val/loss_mae", "val/loss_mlm", "val/loss_contrastive", "val/retrieval/rsum"} <= set(first)
    assert first == second  # masks are reseeded for every evaluation
    assert all(math.isfinite(v) for v in first.values())
    assert trainer.model.training


def test_too_small_training_set_fails_clearly(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco, "data.limit_train=4")
    with Run(cfg, enabled=False) as run, pytest.raises(ValueError, match="no full batch"):
        Trainer(cfg, accelerator, run, MetricLogger(run))
```

`tests/test_train_smoke.py`:
```python
import json

import pytest

from helpers import MODEL_NAMES, run_train


def single_run_dir(cwd):
    runs = sorted((cwd / "res").glob("multimae/default/*"))
    assert len(runs) == 1, runs
    return runs[0]


def check_run(run_dir, two_modalities: bool):
    info = json.loads((run_dir / "run.json").read_text())
    assert info["status"] == "completed"
    assert info["results"]["best_epoch"] == 1
    assert "test/loss" in info["results"]["test"]
    assert ("test/retrieval/rsum" in info["results"]["test"]) == two_modalities
    for name in ("config.yaml", "metrics.jsonl", "train.log", "plots/curves.png"):
        assert (run_dir / name).exists(), name
    return info


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_debug_run_on_cpu(tmp_path, fake_coco, name):
    result = run_train(tmp_path, fake_coco, f"model={name}", "train.save=best")
    assert result.returncode == 0, result.stderr[-5000:]
    run_dir = single_run_dir(tmp_path)
    check_run(run_dir, two_modalities=name.startswith("fusion"))
    assert (run_dir / "checkpoints" / "best.pt").exists()
    # Hydra and wandb write nothing outside the run folder
    assert not (tmp_path / "outputs").exists() and not (tmp_path / ".hydra").exists()
    assert not (tmp_path / "wandb").exists()


def test_train_two_cpu_processes(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "model=fusion_concat", nproc=2)
    assert result.returncode == 0, result.stderr[-5000:]
    info = check_run(single_run_dir(tmp_path), two_modalities=True)  # one folder, not one per rank
    assert info["num_processes"] == 2


@pytest.mark.slow
@pytest.mark.parametrize("name", ["fusion_concat", "fusion_multilearner"])
def test_train_debug_run_on_gpu_real_clip(tmp_path, name):
    import os
    import subprocess
    import sys

    from helpers import REPO

    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_USE_CPU")}
    cmd = [sys.executable, str(REPO / "train.py"), f"model={name}", "train=debug", f"paths.res_dir={tmp_path / 'res'}"]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-5000:]
    check_run(single_run_dir(tmp_path), two_modalities=True)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_trainer.py tests/test_train_smoke.py -q -x`
Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.engine.trainer'`.

- [ ] **Step 5: Implement the trainer**

`mmae/engine/trainer.py`:
```python
"""Training loop: per-epoch validation with retrieval, early stopping, best-weight restore, final test."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mmae.data import CocoPairs, CocoRetrieval, Collator, build_image_transform
from mmae.engine.retrieval import evaluate_retrieval
from mmae.models import MultiMAE
from mmae.utils.logging import MetricLogger
from mmae.utils.run import Run

log = logging.getLogger(__name__)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1 over `warmup_steps`, then cosine decay to 0 at `total_steps`."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EarlyStopper:
    def __init__(self, mode: str, patience: int, min_delta: float) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"monitor mode must be 'max' or 'min', got {mode!r}")
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Record a validation value; True if it is a new best (by more than min_delta)."""
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@dataclass
class EvalLoaders:
    pairs: DataLoader
    retrieval: DataLoader | None


class Trainer:
    def __init__(self, cfg: DictConfig, accelerator: Accelerator, run: Run, metric_logger: MetricLogger) -> None:
        self.cfg, self.accelerator, self.run, self.metric_logger = cfg, accelerator, run, metric_logger
        tcfg, dcfg, mcfg = cfg.train, cfg.data, cfg.model
        self.two_modalities = set(mcfg.modalities) == {"image", "text"}
        self.transform = build_image_transform(mcfg.backbone.processor)
        self.collator = Collator(mcfg.backbone.processor, dcfg.max_text_len)

        model = MultiMAE(mcfg, max_text_len=dcfg.max_text_len)
        train_set = CocoPairs(dcfg.images_dir, dcfg.annotations_dir, "train", self.transform, dcfg.limit_train)
        train_loader = self._loader(train_set, tcfg.batch_size, shuffle=True, drop_last=True, collate=self.collator.pairs)
        optimizer = torch.optim.AdamW(model.param_groups(tcfg.lr, tcfg.lr_backbone, tcfg.weight_decay))
        self.model, self.optimizer, self.train_loader = accelerator.prepare(model, optimizer, train_loader)
        if len(self.train_loader) == 0:
            raise ValueError(
                f"training set of {len(train_set)} items yields no full batch of {tcfg.batch_size} "
                f"per process ({accelerator.num_processes} processes)"
            )
        total_steps = math.ceil(len(self.train_loader) / tcfg.grad_accum) * tcfg.epochs
        # Built on the raw optimizer and stepped by hand once per optimizer step, so it does not depend on
        # accelerate's scheduler stepping rules.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine(step, tcfg.warmup_steps, total_steps)
        )
        self.eval_loaders: dict[str, EvalLoaders] = {}
        self.global_step = 0

    def _loader(self, dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool, collate) -> DataLoader:
        workers = int(self.cfg.train.num_workers)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=collate,
            num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        )

    def _eval_loaders(self, split: str) -> EvalLoaders:
        if split not in self.eval_loaders:
            d, batch = self.cfg.data, self.cfg.train.eval_batch_size
            limit = d.limit_val if split == "val" else d.limit_test
            pairs = self._loader(
                CocoPairs(d.images_dir, d.annotations_dir, split, self.transform, limit),
                batch, shuffle=False, drop_last=False, collate=self.collator.pairs,
            )
            retrieval = None
            if self.two_modalities:
                retrieval = self._loader(
                    CocoRetrieval(d.images_dir, d.annotations_dir, split, self.transform, limit),
                    batch, shuffle=False, drop_last=False, collate=self.collator.retrieval,
                )
                pairs, retrieval = self.accelerator.prepare(pairs, retrieval)
            else:
                pairs = self.accelerator.prepare(pairs)
            self.eval_loaders[split] = EvalLoaders(pairs, retrieval)
        return self.eval_loaders[split]

    def _reduce_means(self, sums: dict[str, float], count: int) -> dict[str, float]:
        keys = sorted(sums)
        values = torch.tensor([sums[k] / max(count, 1) for k in keys], device=self.accelerator.device)
        return dict(zip(keys, self.accelerator.reduce(values, reduction="mean").tolist()))

    def _log_train(self, sums: dict[str, float], count: int, epoch: int) -> None:
        metrics: dict[str, Any] = {f"train/{k}": v for k, v in self._reduce_means(sums, count).items()}
        for group in self.optimizer.param_groups:
            key = "train/lr_backbone" if group["name"].startswith("backbone") else "train/lr"
            metrics.setdefault(key, group["lr"])
        logit_scale = self.accelerator.unwrap_model(self.model).logit_scale
        if logit_scale is not None:
            metrics["train/logit_scale"] = logit_scale.exp().item()
        metrics["epoch"] = epoch
        self.metric_logger.log(metrics, step=self.global_step)

    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        tcfg = self.cfg.train
        sums: dict[str, float] = {}
        count = 0
        progress = tqdm(self.train_loader, desc=f"epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        for batch in progress:
            with self.accelerator.accumulate(self.model):
                out = self.model(batch)
                self.accelerator.backward(out["loss"])
                if self.accelerator.sync_gradients and tcfg.grad_clip:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            for key, value in out.items():
                sums[key] = sums.get(key, 0.0) + value.detach().float().item()
            count += 1
            if self.accelerator.sync_gradients:
                self.scheduler.step()
                self.global_step += 1
                if self.global_step % tcfg.log_every == 0:
                    self._log_train(sums, count, epoch)
                    sums, count = {}, 0
        if count:
            self._log_train(sums, count, epoch)

    @torch.no_grad()
    def evaluate(self, split: str) -> dict[str, float]:
        """Mean losses on the split's pairs (masks reseeded so every call sees the same masks) and, for
        two-modality models, retrieval metrics. Identical on every process."""
        loaders = self._eval_loaders(split)
        self.model.eval()
        sums: dict[str, float] = {}
        count = 0
        cuda = [self.accelerator.device] if self.accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(self.cfg.seed + self.accelerator.process_index)
            for batch in loaders.pairs:
                for key, value in self.model(batch).items():
                    sums[key] = sums.get(key, 0.0) + value.float().item()
                count += 1
        metrics = {f"{split}/{k}": v for k, v in self._reduce_means(sums, count).items()}
        if loaders.retrieval is not None:
            retrieval = evaluate_retrieval(self.model, loaders.retrieval, self.accelerator)
            metrics.update({f"{split}/retrieval/{k}": v for k, v in retrieval.items()})
        self.model.train()
        return metrics

    def _save_best(self, epoch: int, metrics: dict[str, float]) -> None:
        if self.cfg.train.save != "best":
            return
        state = self.accelerator.unwrap_model(self.model).state_dict()
        self.run.save_checkpoint(
            "best.pt",
            {"model": state, "config": OmegaConf.to_container(self.cfg, resolve=True), "epoch": epoch, "metrics": metrics},
        )

    def fit(self) -> dict[str, Any]:
        tcfg, monitor = self.cfg.train, self.cfg.model.monitor
        stopper = EarlyStopper(monitor.mode, tcfg.patience, tcfg.min_delta)
        best_state, best_epoch, best_metrics = None, None, {}
        for epoch in range(1, tcfg.epochs + 1):
            self.train_epoch(epoch)
            if epoch % tcfg.eval_every != 0 and epoch != tcfg.epochs:
                continue
            metrics = self.evaluate("val")
            self.metric_logger.log({**metrics, "epoch": epoch}, step=self.global_step)
            if monitor.metric not in metrics:
                raise KeyError(f"monitor metric {monitor.metric!r} not in {sorted(metrics)}")
            if stopper.update(metrics[monitor.metric]):
                best_epoch, best_metrics = epoch, metrics
                state = self.accelerator.unwrap_model(self.model).state_dict()
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
                self._save_best(epoch, metrics)
                log.info("epoch %d: new best %s = %.4f", epoch, monitor.metric, metrics[monitor.metric])
            elif stopper.should_stop:
                log.info("early stopping after epoch %d", epoch)
                break
        if best_state is not None:
            self.accelerator.unwrap_model(self.model).load_state_dict(best_state)
            log.info("restored the best weights (epoch %s) for the test", best_epoch)
        test = self.evaluate("test")
        self.metric_logger.log({**test, "epoch": best_epoch}, step=self.global_step)
        results = {"best_epoch": best_epoch, "best_val": best_metrics, "test": test}
        self.run.update(results=results)
        return results
```

`mmae/engine/__init__.py`:
```python
"""Training loop and retrieval evaluation."""
from mmae.engine.retrieval import evaluate_retrieval, retrieval_metrics
from mmae.engine.trainer import Trainer

__all__ = ["Trainer", "evaluate_retrieval", "retrieval_metrics"]
```

- [ ] **Step 6: Implement `train.py`**

`train.py`:
```python
"""Train a MultiMAE model.

  python train.py model=fusion_concat                                 # one GPU
  accelerate launch --num_processes 4 --multi_gpu train.py ...        # one node, several GPUs
  python train.py train=debug                                         # tiny smoke run, wandb off
"""
import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig

from mmae.engine.trainer import Trainer
from mmae.utils.logging import MetricLogger, init_wandb, setup_logging
from mmae.utils.run import Run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        mixed_precision=cfg.train.precision, gradient_accumulation_steps=cfg.train.grad_accum
    )
    setup_logging(accelerator.is_main_process)
    set_seed(cfg.seed, device_specific=True)  # different masks per process; DDP syncs the initial weights
    with Run(cfg, enabled=accelerator.is_main_process) as run:
        wandb_run = init_wandb(cfg, run) if accelerator.is_main_process else None
        try:
            Trainer(cfg, accelerator, run, MetricLogger(run, wandb_run)).fit()
        finally:
            if wandb_run is not None:
                wandb_run.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_trainer.py tests/test_train_smoke.py -q && $PY -m pytest -q`
Expected: all pass. If Hydra still writes `outputs/` or a `.log` file in the cwd, check that `override hydra/job_logging: none` and `hydra.output_subdir: null` took effect (`$PY train.py --cfg hydra | grep -A3 job_logging`) before changing anything else.

Then the slow GPU smoke: `$PY -m pytest -m slow tests/test_train_smoke.py -q` (real B/32 on real COCO, `train=debug`). Expected: pass.

- [ ] **Step 8: Confirm the guards bite, then commit**

Temporarily remove the `load_state_dict(best_state)` line: `test_fit_restores_best_weights_before_test` must FAIL. Restore. Temporarily remove `torch.manual_seed(...)` in `evaluate`: `test_evaluate_reports_losses_and_retrieval` must FAIL (`first != second`). Restore.

```bash
git add mmae/engine train.py configs/train/debug.yaml configs/data/coco_cluster.yaml tests/test_trainer.py tests/test_train_smoke.py tests/helpers.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add the trainer and train.py" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 11: `evaluate.py` and the zero-shot known-answer check

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `MultiMAE` (Task 6), `CocoRetrieval`, `Collator`, `build_image_transform` (Task 7), `evaluate_retrieval` (Task 8), `setup_logging`, `update_run_json` (Task 9), `helpers.run_train` (Task 10); config keys `eval.run_dir`, `eval.split`.
- Produces: `evaluate.py` CLI. With `eval.run_dir=<run folder>` it loads `config.yaml` and `checkpoints/best.pt` from there, evaluates retrieval on `eval.split` with the current `data.*` paths, logs the metrics and writes them to `run.json` under `eval.<split>`. Without a run dir it evaluates the pretrained backbone of the current `model` config (zero-shot). Exits with an error for a single-modality model.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:
```python
import json

import pytest

from helpers import REPO, run_train


def test_evaluate_a_trained_run(tmp_path, fake_coco):
    trained = run_train(tmp_path, fake_coco, "model=fusion_concat", "train.save=best")
    assert trained.returncode == 0, trained.stderr[-5000:]
    run_dir = next((tmp_path / "res").glob("multimae/default/*"))
    result = run_train(tmp_path, fake_coco, f"eval.run_dir={run_dir}", "eval.split=val", script="evaluate.py")
    assert result.returncode == 0, result.stderr[-5000:]
    info = json.loads((run_dir / "run.json").read_text())
    assert set(info["eval"]["val"]) >= {"i2t_R1", "t2i_R1", "rsum"}
    assert info["status"] == "completed"  # evaluation does not touch the training status


def test_evaluate_zero_shot_tiny(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "eval.split=test", script="evaluate.py")
    assert result.returncode == 0, result.stderr[-5000:]
    assert "rsum" in result.stderr + result.stdout


def test_evaluate_rejects_single_modality(tmp_path, fake_coco):
    result = run_train(tmp_path, fake_coco, "model=image_mae", script="evaluate.py")
    assert result.returncode != 0 and "both modalities" in result.stderr


@pytest.mark.slow
def test_zero_shot_clip_b32_matches_published(tmp_path):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "ACCELERATE_USE_CPU")}
    out = tmp_path / "zero_shot.json"
    cmd = [sys.executable, str(REPO / "evaluate.py"), "eval.split=test", "train.eval_batch_size=256", f"eval.output={out}"]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-5000:]
    metrics = json.loads(out.read_text())
    # OpenAI CLIP ViT-B/32 zero-shot, COCO 5k test: i2t R@1 50.1, t2i R@1 30.4 (published)
    assert abs(metrics["i2t_R1"] - 50.1) < 1.5, metrics
    assert abs(metrics["t2i_R1"] - 30.4) < 1.5, metrics
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_evaluate.py -q`
Expected: FAIL (`evaluate.py` does not exist: non-zero return code, "can't open file").

- [ ] **Step 3: Implement**

`evaluate.py`:
```python
"""Retrieval evaluation of a trained run or of the zero-shot pretrained backbone.

  python evaluate.py eval.run_dir=res/multimae/default/20261001_120000_fusion_concat   # a trained run
  python evaluate.py model=fusion_concat eval.split=test                               # zero-shot CLIP
"""
import json
import logging
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from mmae.data import CocoRetrieval, Collator, build_image_transform
from mmae.engine.retrieval import evaluate_retrieval
from mmae.models import MultiMAE
from mmae.utils.logging import setup_logging
from mmae.utils.run import update_run_json

log = logging.getLogger("evaluate")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(mixed_precision=cfg.train.precision)
    setup_logging(accelerator.is_main_process)
    split = cfg.eval.split
    run_dir = Path(cfg.eval.run_dir) if cfg.eval.run_dir else None
    if run_dir is not None:
        run_cfg = OmegaConf.load(run_dir / "config.yaml")
        model_cfg, max_text_len = run_cfg.model, run_cfg.data.max_text_len
    else:
        model_cfg, max_text_len = cfg.model, cfg.data.max_text_len
    if set(model_cfg.modalities) != {"image", "text"}:
        raise SystemExit(f"retrieval needs a model with both modalities; {model_cfg.name} has {list(model_cfg.modalities)}")

    model = MultiMAE(model_cfg, max_text_len=max_text_len)
    if run_dir is not None:
        checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        log.info("loaded %s (epoch %s)", run_dir / "checkpoints" / "best.pt", checkpoint.get("epoch"))
    else:
        log.info("zero-shot evaluation of %s", model_cfg.backbone.pretrained)

    data = cfg.data
    dataset = CocoRetrieval(
        data.images_dir, data.annotations_dir, split, build_image_transform(model_cfg.backbone.processor),
        data.limit_val if split == "val" else data.limit_test,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.train.eval_batch_size, shuffle=False, num_workers=cfg.train.num_workers,
        collate_fn=Collator(model_cfg.backbone.processor, max_text_len).retrieval,
    )
    model, loader = accelerator.prepare(model, loader)
    metrics = evaluate_retrieval(model, loader, accelerator)
    if accelerator.is_main_process:
        log.info("%s retrieval on %d images: %s", split, len(dataset),
                 ", ".join(f"{k}={v:.2f}" for k, v in metrics.items()))
        if run_dir is not None:
            info = json.loads((run_dir / "run.json").read_text())
            update_run_json(run_dir, eval={**info.get("eval", {}), split: metrics})
        if cfg.eval.output:
            Path(cfg.eval.output).write_text(json.dumps(metrics, indent=2))
    accelerator.end_training()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_evaluate.py -q && $PY -m pytest -q`
Expected: all pass.

Then the known-answer check on the GPU: `$PY -m pytest -m slow tests/test_evaluate.py -q -s`. Expected: pass, with i2t R@1 within 1.5 of 50.1 and t2i R@1 within 1.5 of 30.4. Record the exact numbers (they go into the Task 12 report). If it fails, do not loosen the tolerance: compare our transform against OpenAI's (`torchvision` resize/crop, mean/std), check that captions are not truncated more than CLIP's 77 tokens would (`data.max_text_len=77` changes the numbers by how much?), and check the EOS pooling; write the investigation into `tests/20261001_zero_shot_check/20261001_zero_shot_check_log.md`.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add evaluate.py with the zero-shot CLIP known-answer check" -m "<paste the measured zero-shot i2t/t2i R@1 here>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```

---

### Task 12: Launch scripts, README, CLAUDE.md and the v1 report

**Files:**
- Create: `scripts/run_local.sh`, `scripts/run_cluster.sh`, `docs/reports/auto/v1/2026-10-01_refactor.md`
- Modify: `README.md`, `docs/reports/reports_sum.md`, `CLAUDE.md` (gitignored, local only), `.claude/<yyyymmdd>_log.md`

**Interfaces:**
- Consumes: every earlier task; the measured zero-shot numbers from Task 11; the test counts.
- Produces: user-facing docs and launchers.

- [ ] **Step 1: Write the launch scripts**

`scripts/run_local.sh`:
```bash
#!/usr/bin/env bash
# Train on this machine with every visible GPU.
# Usage: scripts/run_local.sh [note] [hydra overrides...]
#   scripts/run_local.sh "first try" model=fusion_multilearner train.epochs=20
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#gpus[@]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
else
  NUM_PROCS=1
fi
[ "${NUM_PROCS:-0}" -lt 1 ] && NUM_PROCS=1

NOTE=${1:-local}
shift || true
MULTI=()
[ "${NUM_PROCS}" -gt 1 ] && MULTI=(--multi_gpu)
echo "Launching on ${NUM_PROCS} process(es)"
accelerate launch --num_processes "${NUM_PROCS}" --num_machines 1 --mixed_precision no --dynamo_backend no \
  "${MULTI[@]}" train.py "wandb.notes='${NOTE}'" "wandb.tags=[local]" "$@"
```

`scripts/run_cluster.sh`:
```bash
#!/usr/bin/env bash
# Train on a cluster node (COCO on the node's local disk), every visible GPU.
# Usage: scripts/run_cluster.sh <batch_size> <epochs> <note> [hydra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 3 ]; then
  echo "usage: $0 <batch_size> <epochs> <note> [hydra overrides...]" >&2
  exit 2
fi
BATCH_SIZE=$1 EPOCHS=$2 NOTE=$3
shift 3

if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCS=${#gpus[@]}
else
  NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
[ "${NUM_PROCS:-0}" -lt 1 ] && NUM_PROCS=1
MULTI=()
[ "${NUM_PROCS}" -gt 1 ] && MULTI=(--multi_gpu)
echo "Launching on ${NUM_PROCS} process(es)"
accelerate launch --num_processes "${NUM_PROCS}" --num_machines 1 --mixed_precision no --dynamo_backend no \
  "${MULTI[@]}" train.py data=coco_cluster "train.batch_size=${BATCH_SIZE}" "train.epochs=${EPOCHS}" \
  "wandb.notes='${NOTE}'" "wandb.tags=[cluster]" "$@"
```

```bash
chmod +x scripts/run_local.sh scripts/run_cluster.sh
bash -n scripts/run_local.sh && bash -n scripts/run_cluster.sh
scripts/run_local.sh smoke train=debug   # must finish with a completed run folder under res/multimae/default/
```

- [ ] **Step 2: Rewrite `README.md`**

Replace the Chinese v0 README with an English one covering, in this order: what MultiMAE is (one paragraph: fusion MAE on CLIP towers, three losses, retrieval benchmark); install (`conda activate MultiMAE`, the torch cu130 line from `requirements.txt`, `pip install -r requirements.txt`, `pip install -e ".[dev]"`); data (Karpathy JSON names and the `data.images_dir` / `data.annotations_dir` keys, `data=coco_cluster`); training (`python train.py model=<name>`, the five model configs in a table with one line each, `train=debug`, `scripts/run_local.sh`, `scripts/run_cluster.sh`, multi-GPU via `accelerate launch`); evaluation (`evaluate.py` both modes, the zero-shot numbers measured in Task 11); outputs (the `res/` run folder tree and `scripts/list_runs.py`); tests (`pytest`, `pytest -m slow`); where the old code is (`legacy-v0` tag, `legacy` branch); docs (`docs/reports/reports_sum.md`, the spec). Log the README change in `.claude/<yyyymmdd>_log.md` (before: v0 Chinese README; after: v1 English README; why).

- [ ] **Step 3: Rewrite the local `CLAUDE.md`**

Keep the standard header. Sections: Environment (`MultiMAE` env and the rebuild commands), Commands (train/evaluate/test/list runs/check reports), Architecture (call path `train.py → Trainer → MultiMAE`; the forward pass in five lines; fusion registry; how to add a fusion type or backbone; the HF-internals warning pointing at `tests/test_backbones.py`), Data, Configs (groups and the `_self_`-first ordering, `train/debug.yaml` repeats every train key), Multi-GPU (prepared eval loaders, per-batch gather, `ACCELERATE_USE_CPU=1 torchrun` for CPU DDP), Run folders, Testing conventions (fast vs slow, `helpers.py`, fake COCO, tiny CLIP, DDP workers), Docs and reports. Remove everything about v0's `src/hook`. `CLAUDE.md` is gitignored, so it is not committed.

- [ ] **Step 4: Write the v1 report and index it**

`docs/reports/auto/v1/2026-10-01_refactor.md`, following the user's report-writing rules (paper-draft style, "we", past tense, no dashes as punctuation, numbers for every claim, a baseline beside every headline number): title; §1 what we started from (v0's problems with evidence: no input masking, frozen towers, random projections, broken entry points, DDP crashes, hard-coded 16 px patches, pad == EOS attention bug, warmup that never finished); §2 what we changed (the architecture with a mermaid diagram of the forward pass, colour legend per the rules: grey same, orange replaced, teal new); §3 how we verified it (table: each test group, what it pins, count of tests; the guard-bites checks; the 2-process checks); §4 the known-answer result: zero-shot CLIP B/32 on COCO 5k test, our measured i2t/t2i R@1/R@5/R@10 beside the published 50.1 / 30.4 R@1; §5 what is not done (LR tuning, augmentation, resume, more backbones, cluster CLI). Add its row to the `auto/v1` table in `docs/reports/reports_sum.md` and point "Start here" at it.

Run: `$PY scripts/check_reports_sum.py`
Expected: `reports_sum.md: OK`.

- [ ] **Step 5: Full test run, then commit**

Run: `$PY -m pytest -q && $PY -m pytest -m slow -q`
Expected: all pass. Put the pass counts in the report §3 if they changed.

```bash
git add scripts/run_local.sh scripts/run_cluster.sh README.md docs/reports
git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit -q -m "Add launch scripts, README and the v1 refactor report" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Xsr5aN7rM2EVWRoZMX1BBX"
```
