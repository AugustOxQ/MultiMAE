# H-b Plan 3: COCO decoder probes D1 (decoder re-ranking on ECCV Caption) and D2 (blend probe)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On existing COCO Stage 2 checkpoints, test whether the caption decoder helps as a prior-corrected re-ranker of the dual encoder (D1) and whether it keeps both readings of an image blend, with partial captions selecting one (D2).

**Architecture:** `mmae/engine/hb/coco_probes.py` holds the shared pieces (ECCV i2t queries and per-query AP@R, dual-encoder top-k, decoder caption scores through `MultiMAE.decode_text`, blends). `scripts/hb_coco_d1.py` and `scripts/hb_coco_d2.py` run one checkpoint list each and write JSON; `scripts/hb_coco_table.py` summarises over seeds.

**Tech Stack:** PyTorch, numpy, `eccv_caption`. Python: `/root/miniconda3/envs/MultiMAE/bin/python`.

**Spec:** `docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md`, sections 10 and 11. Depends on Plan 2 Task 1 (`MultiMAE.decode_text`).

## Global Constraints

- Checkpoints (seeds 42 to 44, run folders under `res/coco/multimae/ml_improve/`): `s2_txt80_mae0` (ML-80), `s2_txt80` (ML-80+MAE), `s2_multilearner` (ML-15), `s2_m1clean_txt80`, `s2_none_txt80`; D2 adds `s2_contrastive` (C) and zero-shot CLIP B/32 (dual encoders only).
- D1: image-to-text only; each ECCV query image's top 50 captions from the same model's dual encoder are re-ranked; ECCV mAP@R and R-Precision on the package's i2t queries, overall and by thirds of R (cuts as in `tests/20261003_ml_improve/stratified_eccv.py`: R <= 15, 16 to 20, >= 21).
- Decoder caption score s(c | x) = mean log p of c's hidden tokens. Parallel score: every real token hidden; masked-source models averaged over K = 8 views of 25% (fixed seeds), M1-clean on the full image, `fusion_none` ignores the image. Training-ratio score: K = 8 random masks at the model's own text ratio (one view per mask for masked-source models), mean over masks.
- Null image: the all-zero tensor after CLIP normalisation. PMI = s(c | x) - alpha s(c | null). Re-ranker (a): PMI alone within the top 50; re-ranker (b): z-scored dual similarity + beta x z-scored PMI (z-scores within each query's 50). alpha in {0, 0.25, 0.5, 0.75, 1}, beta in {0, 0.1, 0.25, 0.5, 1, 2}, tuned on the first 1,000 COCO val images (their own 5 captions as positives, AP@R with R = 5), per checkpoint and score type.
- D2 pairs: 1,000 pairs of distinct COCO test images (seed 0) whose mean cross-caption cosine (zero-shot CLIP B/32 text embeddings of their 5 + 5 captions) is below the 10th percentile of 100,000 random pairs (seed 1). Pixel blends lambda A + (1 - lambda) B on the normalised tensors, lambda in {0.3, 0.4, 0.5, 0.6, 0.7}; patch mix: a 13-patch view with m of A's patches and 13 - m of B's at their own positions (positions drawn per pair, seed 2), m in {3, 5, 7, 9, 11}.
- D2 readouts: dual both-covered@k (k in {5, 10, 20}) over the 25,000 test captions; balance = A's share of the top 5 when A's and B's 10 captions are ranked by the model's score (dual similarity, or the decoder's parallel score); decoder selection index for caption cA (A's first caption) with j in {0, 1, 2, 3} of its content tokens visible: (s_blend - s_B) / (s_A - s_B) where s_img = mean log p of cA's hidden tokens with the same visible tokens on image img (pure A, pure B, or the 0.5 blend), and the same for B's first caption with B's tokens visible.
- Commits with `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit`, attribution lines as in the other plans. Fast tests on CPU with the tiny CLIP; the real-data validation is a `slow` test.

## Review Focus

1. The per-query AP@R must reproduce the package's ECCV i2t mAP@R for the same embeddings (else every D1 number is off): Task 1 slow test.
2. Captions retrieved beyond the top 50 keep their dual-encoder order; R can exceed 50 for a few queries: Task 1 test.
3. A checkpoint without a decoder (C, zero-shot) must be skipped by D1 and give dual-only rows in D2: Task 3 test.
4. Blends must be built on the same normalised tensors the models see: Task 3 test (lambda = 1 reproduces image A exactly).
5. The selection index is undefined when s_A == s_B: such items are dropped and counted: Task 3 test.

---

### Task 1: ECCV queries, per-query AP@R, dual top-k

**Files:** Create `mmae/engine/hb/coco_probes.py` (first part), `tests/test_hb_coco_probes.py`.

**Interfaces (produces):**
- `@dataclass EccvI2T: query_image (Q,) int (row in the 5k test order); positives list[ndarray] (caption rows, image-major i * 5 + j); R (Q,) int` and `eccv_i2t(annotations_dir) -> EccvI2T` (via `mmae.engine.eccv.map_coco_ids`, `retrieval_items`, `eccv_caption.Metrics().eccv_gts["i2t"]`; R counts every listed positive, as the package does).
- `ap_at_r(order (Q, depth) caption rows, positives, R) -> ndarray (Q,)` (AP@R exactly as `stratified_eccv.per_query_ap`).
- `topk_captions(image_emb (N, D), caption_emb (N * 5, D), query_image (Q,), k) -> ndarray (Q, k)` (stable descending sort of cosine).
- `rerank(order (Q, depth), top_scores (Q, k)) -> ndarray (Q, depth)`: reorders the first k columns by `top_scores` (stable, descending), leaves columns k.. unchanged.

- [ ] **Step 1: Tests.**

```python
import numpy as np
import pytest
import torch

from mmae.engine.hb import coco_probes as cp


def test_ap_at_r_known_values():
    order = np.array([[3, 1, 2, 0], [0, 1, 2, 3]])
    positives = [np.array([3, 2]), np.array([2, 3])]
    R = np.array([2, 2])
    np.testing.assert_allclose(cp.ap_at_r(order, positives, R), [0.5, 0.0])


def test_rerank_only_touches_the_top_k():
    order = np.array([[5, 4, 3, 2, 1, 0]])
    out = cp.rerank(order, np.array([[0.1, 0.9, 0.5]]))
    assert out.tolist() == [[4, 3, 5, 2, 1, 0]]


def test_topk_is_a_stable_cosine_sort():
    image = torch.tensor([[1.0, 0.0]])
    captions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    assert cp.topk_captions(image, captions, np.array([0]), 3).tolist() == [[0, 2, 3]]


@pytest.mark.slow
def test_per_query_ap_reproduces_the_package():
    """Mean AP@R over the i2t queries equals eccv/i2t_map_at_r of a Stage 2 run from its saved embeddings."""
    from pathlib import Path
    import json
    emb = torch.load(Path("res/coco/diagnostics/stage2/embeddings/zeroshot.pt"))
    q = cp.eccv_i2t(Path("/data/SSD/coco/annotations"))
    image = torch.nn.functional.normalize(emb["image"].float(), dim=-1)
    caption = torch.nn.functional.normalize(emb["caption"].float().reshape(-1, emb["caption"].shape[-1]), dim=-1)
    order = cp.topk_captions(image, caption, q.query_image, int(q.R.max()))
    ours = 100 * cp.ap_at_r(order, q.positives, q.R).mean()
    # zero-shot B/32 ECCV i2t mAP@R from the Stage 0 diagnostics JSON; read it from res/coco/diagnostics/stage2/diagnostics.json
    reported = json.loads(Path("res/coco/diagnostics/stage2/diagnostics.json").read_text())
    assert any(abs(ours - v) < 0.02 for v in _find_values(reported, "i2t_map_at_r"))
```

(`_find_values` walks the JSON and returns every float under a key containing the name; write it in the test file. If the diagnostics JSON does not hold the zero-shot i2t value, compute the package's value with `mmae.engine.eccv.coco_test_metrics` on the same embeddings instead and compare.)

- [ ] **Step 2: RED; Step 3: implement:**

```python
"""COCO decoder probes for H-b (spec sections 10 and 11): ECCV Caption i2t queries, per-query AP@R, dual-encoder
top-k and re-ranking, decoder caption scores, and image blends."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from eccv_caption import Metrics

from mmae.data.coco import retrieval_items
from mmae.engine.eccv import CAPTIONS_FILE, check_ids, map_coco_ids


@dataclass
class EccvI2T:
    query_image: np.ndarray
    positives: list[np.ndarray]
    R: np.ndarray


def eccv_i2t(annotations_dir: str | Path) -> EccvI2T:
    annotations_dir = Path(annotations_dir)
    items = retrieval_items(annotations_dir, "test")
    image_ids, caption_ids = map_coco_ids(items, annotations_dir / CAPTIONS_FILE)
    metrics = Metrics()
    check_ids(image_ids, caption_ids, metrics)
    image_row = {int(i): n for n, i in enumerate(image_ids)}
    caption_row = {int(c): n for n, c in enumerate(caption_ids.reshape(-1))}
    gts = metrics.eccv_gts["i2t"]
    queries = sorted(gts)
    return EccvI2T(
        query_image=np.array([image_row[int(q)] for q in queries]),
        positives=[np.array(sorted(caption_row[int(p)] for p in set(gts[q]) if int(p) in caption_row)) for q in queries],
        R=np.array([len(set(gts[q])) for q in queries]),
    )


def ap_at_r(order: np.ndarray, positives: list[np.ndarray], R: np.ndarray) -> np.ndarray:
    out = np.zeros(len(R))
    for i, r in enumerate(R):
        rel = np.isin(order[i, :r], positives[i]).astype(float)
        out[i] = (np.cumsum(rel) / np.arange(1, r + 1) * rel).sum() / r
    return out


def topk_captions(image_emb: torch.Tensor, caption_emb: torch.Tensor, query_image: np.ndarray, k: int) -> np.ndarray:
    image = torch.nn.functional.normalize(image_emb.float(), dim=-1)[torch.as_tensor(query_image)]
    captions = torch.nn.functional.normalize(caption_emb.float(), dim=-1)
    out = []
    for start in range(0, len(image), 256):
        scores = image[start : start + 256] @ captions.T
        out.append(torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k])
    return torch.cat(out).numpy()


def rerank(order: np.ndarray, top_scores: np.ndarray) -> np.ndarray:
    k = top_scores.shape[1]
    within = np.argsort(-top_scores, axis=1, kind="stable")
    out = order.copy()
    out[:, :k] = np.take_along_axis(order[:, :k], within, axis=1)
    return out
```

- [ ] **Step 4: GREEN** (fast tests; then `python -m pytest tests/test_hb_coco_probes.py -m slow -k reproduces -q` on CPU). **Step 5: Commit** `hb: ECCV i2t per-query AP@R and re-ranking helpers`.

---

### Task 2: D1 decoder caption scores, PMI tuning and the D1 script

**Files:** Modify `mmae/engine/hb/coco_probes.py`; create `scripts/hb_coco_d1.py`, `scripts/run_hb_coco.sh`; extend `tests/test_hb_coco_probes.py`.

**Interfaces (produces):**
- `caption_scores(model, images (B, 3, 224, 224), captions_tok: dict (B, T) input_ids/attention_mask, kind: "parallel" | "ratio", n_samples: int, seed: int) -> Tensor (B,)`: per pair, mean over samples of the mean log p of the hidden tokens (gathered from `decode_text`'s caption logits at the hidden positions). For "parallel" every real token is hidden and samples differ only in the view (masked-source models; `ids_keep` from `random_patch_mask(B, 49, 0.75, generator=torch.Generator().manual_seed(seed + s))`; clean-source and `fusion_none` models use `ids_keep=None` and one sample). For "ratio" the token mask is `random_token_mask(..., model.text_ratio, generator=...)` with the same per-sample seeds, and the view is drawn as above.
- `null_images(n) -> Tensor (n, 3, 224, 224)` zeros.
- `tune(dual (Q, k), dec (Q, k), null (Q, k), order (Q, k), positives, R) -> dict(alpha_a, alpha_b, beta)` maximising mean AP@R (grids in Global Constraints).
- `scripts/hb_coco_d1.py --runs <run_dir>... --out <dir> [--annotations-dir --images-dir --k 50 --samples 8 --val-images 1000 --batch-size]`: for each decoder checkpoint: dual embeddings of the 5k test and the first 1,000 val images (`evaluate.py`'s path: `CocoRetrieval` + `encode_retrieval_set`), top 50 per query, both decoder scores and the null scores of every (query, candidate) pair, tuning on val, then test AP@R for the dual order, re-ranker (a) and (b) per score type, overall and per R third; writes `<out>/<run>/d1.json`.

- [ ] **Steps:** tests first (CPU, tiny CLIP: `caption_scores` is finite, shape (B,), "parallel" with `fusion_none` ignores the image (equal scores for two different images), "ratio" differs between two seeds; `tune` picks alpha = 0 and beta = 0 when the decoder scores are noise and the dual order is perfect); implement; `run_hb_coco.sh` is the cluster launch form (`exec python scripts/hb_coco_$1.py "${@:2}"`); a CPU script smoke on the fake COCO is not possible for ECCV (it needs the real 5k test), so the script gets a `--limit-queries` option and is smoke-tested on the real data on GPU in Task 4. Commit `hb: D1 decoder re-ranking on ECCV Caption`.

---

### Task 3: D2 blend probe

**Files:** Modify `mmae/engine/hb/coco_probes.py`; create `scripts/hb_coco_d2.py`; extend `tests/test_hb_coco_probes.py`.

**Interfaces (produces):**
- `select_pairs(caption_emb (N, 5, D), n_pairs=1000, n_random=100000, percentile=10, seed_pairs=0, seed_random=1) -> ndarray (n_pairs, 2)`.
- `blend(a, b, lam) -> Tensor` (= lam a + (1 - lam) b on normalised tensors); `patch_mix(a, b, m, generator) -> tuple[Tensor composite, Tensor ids_keep (13,)]` (13 positions drawn, the first m from A).
- `both_covered(order (P, k) caption rows, a_rows (P, 5), b_rows (P, 5)) -> float`; `balance(scores_a (P, 5), scores_b (P, 5)) -> ndarray (P,)` (A's share of the top 5 of the 10).
- `selection_index(s_blend, s_a, s_b) -> tuple[ndarray, int]` (index per item and the number dropped where s_a == s_b).
- `scripts/hb_coco_d2.py --runs <run_dir>... [--zero-shot] --out <dir>`: per model, dual readouts for every lambda and m; decoder readouts (balance by parallel score, both-covered@k after D1's re-ranker (a) with that checkpoint's tuned alpha read from `--d1-dir`, selection index for j in {0, 1, 2, 3}); writes `<out>/<run>/d2.json`.

- [ ] **Steps:** tests first (`blend(a, b, 1.0)` equals `a` exactly; `patch_mix` composite holds A's pixels on m of the 13 kept patches and B's on the rest; `both_covered` and `balance` on hand-made orders; `selection_index` drops ties and is 1.0 when s_blend == s_a; a contrastive model gives dual rows only); implement; commit `hb: D2 blend probe`.

---

### Task 4 (controller): run D1 and D2

- [ ] Build the input folder as for the Stage 2 diagnostics (hard links of the needed run folders, `tests/20261003_ml_improve/build_stage2_diag_input.sh` pattern; add `s2_txt80_mae0`), check the local GPU or use `cluster launch -- bash scripts/run_hb_coco.sh d1 ...` on a node with the data, then D2 (reads D1's tuned alpha), then `scripts/hb_coco_table.py` (means over seeds with seed intervals; the review's kill criteria for D1 and D2 stated beside the numbers). Log results in `tests/20261007_hb/runs.md`.
