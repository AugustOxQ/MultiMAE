# H-b Plan 2: D4, D6 and D7 readouts (emotion distributions, calibration, baselines, decision rule)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the D4 checkpoints, compute every model's and baseline's emotion distribution on the AL-28 paintings, fit temperatures, score them against the dense human votes, apply the pre-registered decision rule, and produce the D6 (view sampling) and D7 (partial caption) tables; plus the gate check that precedes wave 2.

**Architecture:** Two stages. A GPU stage (`scripts/hb_encode.py`) runs only forward passes and writes one `encode.npz` per run: decoder emotion logits per view and caption length, pooled image embeddings (full image and per view), prompt embeddings, train-painting embeddings, and D7 logits. A CPU stage (`scripts/hb_table.py`, `scripts/hb_gate.py`) fits temperatures and probes, computes metrics and the hierarchical bootstrap, and writes JSON and markdown tables. Library code lives in `mmae/engine/hb/` (one file per job); a new `MultiMAE.decode_text` exposes the caption decoder for given masks.

**Tech Stack:** PyTorch, numpy, scipy, pandas. Python: `/root/miniconda3/envs/MultiMAE/bin/python`.

**Spec:** `docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md` (sections 6 to 9 and 13). Plan 1 (`docs/superpowers/plans/2026-10-07-hb-1-training-path.md`) built the model and data this plan reads.

## Global Constraints

- Emotion order: `mmae.data.artelingo.EMOTIONS`; AL-28 "other" merges into "something else".
- Dense target: per AL-28 painting, the 9-class histogram of its non-English votes; paintings with fewer than 20 such votes are dropped (1,600 of 1,658 remain).
- Painting-level decoder readout: every caption position hidden; average of softmax(logits / T) over caption lengths at the 10th, 20th, ..., 90th percentiles of real non-special token counts in ArtELingo train (held-out paintings excluded, truncated at `max_text_len - 2`), and over K = 16 random 25% views for ML-80 and ML-80+MAE (in-distribution readout) or the full clean image for Par-cap.
- Temperature: one scalar per (arm, seed, readout), minimising the mean NLL of individual ArtELingo validation labels (held-out paintings excluded), log T in [-3, 3]; applied to logits before averaging; every baseline gets the same fit.
- Prompts: `"a painting that evokes {label}."` for each of the 9 labels; logits = exp(logit_scale) x cosine.
- Soft-label probe: multinomial logistic regression on frozen L2-normalised pooled image embeddings, soft targets = per-painting train label histograms (normalised), weight decay chosen on validation NLL from {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}; primary comparator = per seed, the probe with the lowest validation NLL among the four arms' embeddings.
- Metrics: JSD = Jensen-Shannon distance, base 2; entropy Spearman over paintings; KL(human || model); TVD; RankCS = mean over paintings of the Spearman correlation between the model's 9 probabilities and the human proportions (paintings with a constant human vector skipped).
- Decision rule (spec section 7): four tests (JSD, entropy Spearman) x (vs Par-cap, vs strongest probe) for ML-80; hierarchical bootstrap B = 10,000 (seeds per arm resampled independently, then paintings shared across arms); two-sided p = 2 min(P(diff <= 0), P(diff >= 0)); Holm over the four; support if for at least one metric both comparisons favour ML-80 with Holm p < 0.05.
- Gate (spec section 13): readout sums to 1; fitted T in [0.25, 4]; validation NLL below prior-only's; mean JSD between the model's AL-28 distributions and the prior-only distribution > 0.02.
- D7: j in {0, 1, 2, 4, 8} visible content tokens, prefix and random-subset patterns, ML arms averaged over K = 4 views, Par-cap on the full image; null image = all-zero tensor after normalisation.
- Fixed seeds: views and D7 random subsets use `torch.Generator().manual_seed(...)` with the seeds written below, so every arm sees the same views.
- Fast tests: CPU, tiny CLIP, no `-n` (pytest-xdist is not installed). Commits with `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit`, attribution lines `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01ApMQ4krinxjrUg4cqirWE2`. Edits to existing source files go in `.claude/20261007_log.md`.

## Review Focus

1. `decode_text` must reproduce the training forward's caption-decoder outputs exactly for the same masks, for both `mlm_image_source=masked` and `clean`; otherwise every readout measures a different model: Task 1 test.
2. A run folder of the contrastive arm (no decoder) must flow through encode and table without decoder arrays: Tasks 4 and 7 test it.
3. Paintings whose images are grayscale, CMYK or huge must load (WikiArt): handled by Plan 1's `load_painting`; Task 3 reuses it.
4. Temperature fitting with labels that put zero model mass nowhere (softmax) but human zeros everywhere: JSD and KL(human || model) must stay finite: Task 2 tests.
5. The bootstrap must pair paintings across arms (same painting sample) while resampling seeds independently per arm: Task 6 test.

---

### Task 1: `MultiMAE.decode_text` (caption decoder for given masks)

**Files:**
- Modify: `mmae/models/model.py` (new method), `tests/test_model.py` (`recorder` also records `ids_keep`)
- Create: `tests/test_decode_text.py`

**Interfaces:**
- Produces: `MultiMAE.decode_text(pixel_values, input_ids, attention_mask, token_mask, ids_keep=None) -> tuple[Tensor (B, T, V), Tensor (B, 9) | None]`.

- [ ] **Step 1: Recorder.** In `tests/test_model.py::recorder`, also record the patch ids: in `wrapped`, for the patch mask store `masks["ids_keep"] = result[0]` beside `masks["patch"] = result[1]`.

- [ ] **Step 2: Failing tests** `tests/test_decode_text.py`:

```python
"""MultiMAE.decode_text reproduces forward's caption decoder for the same masks (H-b readouts, plan 2 task 1)."""
import pytest
import torch

from helpers import add_emotion, make_batch
from test_model import recorder, tiny_model

ARMS = {
    "ml80": ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0"),
    "parcap": ("model.emotion_head=true", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean",
               "model.loss.weights.mae=0"),
    "coco_ml15": (),
}


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_decode_text_matches_forward(arm, tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", *ARMS[arm]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    if arm != "coco_ml15":
        batch = add_emotion(batch)
    captured, _, _ = run(batch)
    ids_keep = None if arm == "parcap" else masks["ids_keep"]
    with torch.no_grad():
        logits, emotion = model.decode_text(batch["pixel_values"], batch["input_ids"], batch["attention_mask"],
                                            masks["token"], ids_keep=ids_keep)
    torch.testing.assert_close(logits, captured["text_decoder"], rtol=0, atol=1e-5)
    if arm == "coco_ml15":
        assert emotion is None
    else:
        torch.testing.assert_close(emotion, captured["text_decoder_prefix"][:, 0], rtol=0, atol=1e-5)


def test_decode_text_rejects_models_without_a_text_decoder():
    with pytest.raises(ValueError, match="decode_text"):
        tiny_model("contrastive").decode_text(None, None, None, None)
```

- [ ] **Step 3: Run** `python -m pytest tests/test_decode_text.py -q`. Expected: FAIL (`AttributeError: ... decode_text`).

- [ ] **Step 4: Implement** in `MultiMAE` (after `embed_text`):

```python
    def decode_text(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        token_mask: torch.Tensor, ids_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The caption decoder's outputs for given masks, as in forward's masked pass: the decoder reads the fusion
        of the image patches in ids_keep (all patches if None, whatever mlm_image_source is) and the caption with
        token_mask hidden. Returns (caption logits (B, T, vocab), emotion logits (B, 9) or None). For the H-b
        readouts (spec 2026-10-07, section 6.1); forward is unchanged."""
        if not (self.reconstruction and self.use_image and self.use_text) or self.pooled_conditioning:
            raise ValueError("decode_text needs a two-modality reconstruction model without pooled conditioning")
        image_tokens = self.image_proj(self.vision.encode(pixel_values, ids_keep))
        text_tokens = self.text_proj(self.text.encode(input_ids, attention_mask, token_mask))
        text_padding = ~attention_mask.bool()
        fused = self.fusion(image_tokens, text_tokens, text_padding)
        decoded = self.text_decoder(fused.text_memory, fused.text_padding, query_padding=text_padding)
        if self.emotion_head:
            logits, emotion = decoded
            return logits, emotion[:, 0]
        return decoded, None
```

- [ ] **Step 5: Run** `python -m pytest tests/test_decode_text.py tests/test_model.py tests/test_model_variants.py -q`. Expected: PASS.

- [ ] **Step 6: Guard bites.** In `decode_text`, swap `ids_keep` for `None` for the masked source; `test_decode_text_matches_forward[ml80]` must fail. Revert.

- [ ] **Step 7: Change log; commit** `model: decode_text for the H-b readouts`.

---

### Task 2: Metrics and temperature (`mmae/engine/hb/metrics.py`, `calibrate.py`)

**Files:**
- Create: `mmae/engine/hb/__init__.py` (docstring only), `mmae/engine/hb/metrics.py`, `mmae/engine/hb/calibrate.py`, `tests/test_hb_metrics.py`

**Interfaces:**
- Produces (numpy, float64 inside):
  - `metrics.js_distance(p, q) -> ndarray (N,)` (base 2), `entropy_bits(p) -> (N,)`, `kl(h, m, eps=1e-12) -> (N,)` (KL(h || m), nats), `tvd(p, q) -> (N,)`, `entropy_spearman(p, h) -> float`, `rank_cs(p, h) -> float`, `normalise(counts) -> (N, 9)`.
  - `calibrate.mixture(logits (N, M, 9), temperature) -> (N, 9)`; `calibrate.nll(probs (P, 9), painting_index (n,), labels (n,)) -> float`; `calibrate.fit_temperature(logits (P, M, 9), painting_index, labels) -> float`.

- [ ] **Step 1: Failing tests** `tests/test_hb_metrics.py`:

```python
import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, spearmanr

from mmae.engine.hb import calibrate, metrics

rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(9), size=50)
H = rng.dirichlet(np.ones(9) * 0.5, size=50)
H[0] = [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]  # human zeros


def test_js_distance_matches_scipy():
    expected = np.array([jensenshannon(p, h, base=2) for p, h in zip(P, H)])
    np.testing.assert_allclose(metrics.js_distance(P, H), expected, atol=1e-10)
    assert np.all(metrics.js_distance(P, H) <= 1.0)


def test_entropy_kl_tvd():
    np.testing.assert_allclose(metrics.entropy_bits(P), entropy(P, base=2, axis=1), atol=1e-10)
    np.testing.assert_allclose(metrics.kl(H, P), np.array([entropy(h, p) for h, p in zip(H, P)]), atol=1e-9)
    assert np.isfinite(metrics.kl(H, P)).all()
    np.testing.assert_allclose(metrics.tvd(P, H), 0.5 * np.abs(P - H).sum(1), atol=1e-12)


def test_spearman_metrics():
    rho = spearmanr(entropy(P, axis=1), entropy(H, axis=1)).correlation
    assert metrics.entropy_spearman(P, H) == pytest.approx(rho)
    flat = H.copy()
    flat[1] = 1 / 9  # constant human vector: skipped
    per = [spearmanr(p, h).correlation for i, (p, h) in enumerate(zip(P, flat)) if i != 1]
    assert metrics.rank_cs(P, flat) == pytest.approx(np.mean(per))


def test_normalise_counts():
    counts = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 5]])
    np.testing.assert_allclose(metrics.normalise(counts).sum(1), 1.0)


def test_fit_temperature_recovers_a_known_temperature():
    true_t = 2.0
    logits = rng.normal(size=(400, 3, 9)) * 3
    probs = calibrate.mixture(logits, true_t)
    labels_per = 40
    painting = np.repeat(np.arange(400), labels_per)
    labels = np.array([rng.choice(9, p=probs[i]) for i in painting])
    fitted = calibrate.fit_temperature(logits, painting, labels)
    assert fitted == pytest.approx(true_t, rel=0.1)
    assert calibrate.nll(calibrate.mixture(logits, fitted), painting, labels) <= \
        calibrate.nll(calibrate.mixture(logits, 1.0), painting, labels)


def test_mixture_averages_probabilities_not_logits():
    logits = np.array([[[10.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 10.0, 0, 0, 0, 0, 0, 0, 0]]])
    mixed = calibrate.mixture(logits, 1.0)
    np.testing.assert_allclose(mixed.sum(1), 1.0)
    assert mixed[0, 0] == pytest.approx(mixed[0, 1]) and mixed[0, 0] > 0.49
```

- [ ] **Step 2: Run** `python -m pytest tests/test_hb_metrics.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement.** `mmae/engine/hb/__init__.py`:

```python
"""H-b readouts (spec docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md, sections 6 to 9)."""
```

`mmae/engine/hb/metrics.py`:

```python
"""Distribution agreement metrics between model and human emotion distributions (spec section 7). Inputs are
(N, 9) arrays of probabilities, one row per painting."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def normalise(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def _plogp_terms(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """sum p log2(p / q) per row with 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / q), 0.0)
    return terms.sum(axis=1)


def js_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance, base 2 (in [0, 1]), per row."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    divergence = 0.5 * _plogp_terms(p, m) + 0.5 * _plogp_terms(q, m)
    return np.sqrt(np.clip(divergence, 0.0, None))


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)


def kl(h: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(human || model) in nats per row; the model side is clipped at eps (softmax outputs are never zero)."""
    h, m = np.asarray(h, np.float64), np.clip(np.asarray(m, np.float64), eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0, h * np.log(h / m), 0.0).sum(axis=1)


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(p, np.float64) - np.asarray(q, np.float64)).sum(axis=1)


def entropy_spearman(p: np.ndarray, h: np.ndarray) -> float:
    """Spearman correlation over paintings between model entropy and human entropy."""
    return float(spearmanr(entropy_bits(p), entropy_bits(h)).correlation)


def rank_cs(p: np.ndarray, h: np.ndarray) -> float:
    """Mean over paintings of the Spearman correlation between the model's and the human class proportions;
    paintings whose human vector is constant are skipped."""
    p, h = np.asarray(p, np.float64), np.asarray(h, np.float64)
    keep = h.max(axis=1) > h.min(axis=1)
    rp, rh = rankdata(p[keep], axis=1), rankdata(h[keep], axis=1)
    rp, rh = rp - rp.mean(1, keepdims=True), rh - rh.mean(1, keepdims=True)
    rho = (rp * rh).sum(1) / np.sqrt((rp**2).sum(1) * (rh**2).sum(1))
    return float(rho.mean())
```

`mmae/engine/hb/calibrate.py`:

```python
"""Temperature scaling for the emotion readouts (spec section 6.2): one temperature per (arm, seed, readout),
minimising the mean NLL of individual validation labels; applied to logits before averaging over views and
caption lengths."""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, softmax

LOG_T_GRID = np.linspace(-3.0, 3.0, 241)


def mixture(logits: np.ndarray, temperature: float) -> np.ndarray:
    """(N, M, 9) logits over M samples (views x lengths) -> (N, 9) mean of softmax(logits / T)."""
    return softmax(np.asarray(logits, np.float64) / temperature, axis=-1).mean(axis=1)


def nll(probs: np.ndarray, painting_index: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of individual labels; label i belongs to painting painting_index[i]."""
    picked = probs[painting_index, labels]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, painting_index: np.ndarray, labels: np.ndarray) -> float:
    """Grid search over log T in [-3, 3] (step 0.025), then a local refinement to 1e-4 in log T."""
    def loss(log_t: float) -> float:
        return nll(mixture(logits, float(np.exp(log_t))), painting_index, labels)

    losses = [loss(x) for x in LOG_T_GRID]
    best = int(np.argmin(losses))
    lo, hi = LOG_T_GRID[max(best - 1, 0)], LOG_T_GRID[min(best + 1, len(LOG_T_GRID) - 1)]
    for _ in range(40):  # golden-section search on [lo, hi]
        a, b = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        if loss(a) < loss(b):
            hi = b
        else:
            lo = a
        if hi - lo < 1e-4:
            break
    return float(np.exp(0.5 * (lo + hi)))
```

- [ ] **Step 4: Run** `python -m pytest tests/test_hb_metrics.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** `hb: agreement metrics and temperature fitting`.

---

### Task 3: Readout data (`mmae/engine/hb/data.py`)

**Files:**
- Create: `mmae/engine/hb/data.py`, `tests/test_hb_data.py`
- Modify: `tests/helpers.py` (`make_fake_al28`)

**Interfaces:**
- Consumes: `mmae.data.artelingo` (`EMOTIONS`, `EMOTION_INDEX`, `heldout_paintings`, `read_json`, `SPLIT_FILES`).
- Produces:
  - `AL28_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"`.
  - `@dataclass Paintings: names: list[str]; images: list[str]` (relative image paths under the WikiArt root).
  - `al28_targets(csv, min_votes=20) -> tuple[Paintings, ndarray (N, 9) counts]` (non-English votes, "other" merged).
  - `english_counts(names, annotations_dir) -> ndarray (N, 9)` (English labels from the train, val and test files).
  - `val_labels(annotations_dir, heldout) -> tuple[Paintings, ndarray painting_index (n,), ndarray labels (n,)]`.
  - `train_histograms(annotations_dir, heldout) -> tuple[Paintings, ndarray (N, 9) counts]`.
  - `prior(annotations_dir, heldout) -> ndarray (9,)`.
  - `length_grid(annotations_dir, tokenizer, heldout, max_text_len) -> list[int]` (9 lengths).
  - `test_captions(annotations_dir) -> list[dict]` (keys `painting`, `image`, `caption`, `emotion` index).
  - `tests/helpers.make_fake_al28(path, paintings) -> Path` (a CSV in the AL-28 format).

- [ ] **Step 1: Fake AL-28** in `tests/helpers.py`:

```python
def make_fake_al28(path: Path, paintings: dict[str, str]) -> Path:
    """An AL-28-format CSV: painting -> image_name; votes in three languages incl. English and one 'other'."""
    import csv

    rows = []
    for k, (painting, image) in enumerate(paintings.items()):
        for v in range(25):
            emotion = "other" if v == 1 else ARTELINGO_EMOTIONS[(k + v) % 3]  # v == 1 is a Hausa vote
            rows.append({"image_id": float(k), "genre": "g", "emotion": emotion, "caption": "c", "art_style": "Style_A",
                         "painting": painting, "language": ("english", "Hausa", "Thai")[v % 3], "image_name": image,
                         "split": "train"})
    rows += [{"image_id": 99.0, "genre": "g", "emotion": "awe", "caption": "c", "art_style": "Style_A",
              "painting": "sparse", "language": "Thai", "image_name": "Style_A/sparse.jpg", "split": "train"}] * 5
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
```

- [ ] **Step 2: Failing tests** `tests/test_hb_data.py`:

```python
import numpy as np

from helpers import make_fake_al28
from mmae.data.artelingo import EMOTIONS, heldout_paintings
from mmae.engine.hb import data


def test_al28_targets_drop_english_merge_other_and_filter(tmp_path):
    csv = make_fake_al28(tmp_path / "al28.csv", {"p5": "Style_A/p5.jpg", "v2": "Style_A/v2.jpg"})
    paintings, counts = data.al28_targets(csv, min_votes=20)
    assert paintings.names == ["p5", "v2"] and paintings.images == ["Style_A/p5.jpg", "Style_A/v2.jpg"]
    # 25 votes each, every third English (v = 0, 3, ..., 24: 9 votes) -> 16 non-English; "sparse" has 5 -> dropped
    assert counts.sum(1).tolist() == [16, 16]
    assert counts[:, EMOTIONS.index("something else")].tolist() == [1, 1]  # the merged "other" vote


def test_val_labels_and_train_histograms_respect_the_holdout(fake_artelingo):
    _, annotations, heldout_file = fake_artelingo
    heldout = heldout_paintings(heldout_file)
    paintings, index, labels = data.val_labels(annotations, heldout)
    assert "v2" not in paintings.names and len(index) == len(labels) == 10
    train, counts = data.train_histograms(annotations, heldout)
    assert "p5" not in train.names and counts.sum() == 10
    prior = data.prior(annotations, heldout)
    assert prior.shape == (9,) and np.isclose(prior.sum(), 1.0)


def test_english_counts(fake_artelingo):
    _, annotations, _ = fake_artelingo
    counts = data.english_counts(["p1", "t0", "missing"], annotations)
    assert counts.sum(1).tolist() == [2, 5, 0]


def test_length_grid(fake_artelingo, tokenizer):
    _, annotations, heldout_file = fake_artelingo
    grid = data.length_grid(annotations, tokenizer, heldout_paintings(heldout_file), max_text_len=40)
    assert len(grid) == 9 and all(isinstance(x, int) and 1 <= x <= 38 for x in grid)
    assert grid == sorted(grid)


def test_test_captions(fake_artelingo):
    _, annotations, _ = fake_artelingo
    caps = data.test_captions(annotations)
    assert len(caps) == 15 and set(caps[0]) == {"painting", "image", "caption", "emotion"}
```

- [ ] **Step 3: Run** `python -m pytest tests/test_hb_data.py -q`. Expected: FAIL (module missing).

- [ ] **Step 4: Implement** `mmae/engine/hb/data.py`:

```python
"""Data for the H-b readouts (spec sections 4, 6 and 9): the AL-28 dense targets, English reference counts,
validation labels for temperature fitting, train histograms for the probe and the prior, the caption-length grid,
and the D7 test captions."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmae.data.artelingo import EMOTION_INDEX, SPLIT_FILES, read_json

AL28_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"
AL28_MERGE = {"other": "something else"}


@dataclass
class Paintings:
    names: list[str]
    images: list[str]


def _histograms(items: list[dict], keep) -> tuple[Paintings, np.ndarray]:
    counts: dict[str, np.ndarray] = {}
    images: dict[str, str] = {}
    for item in items:
        if not keep(item["painting"]):
            continue
        row = counts.setdefault(item["painting"], np.zeros(9, dtype=np.int64))
        row[EMOTION_INDEX[item["emotion"]]] += 1
        images.setdefault(item["painting"], item["image"])
    names = sorted(counts)
    return Paintings(names, [images[n] for n in names]), np.stack([counts[n] for n in names]) if names else np.zeros((0, 9))


def al28_targets(csv: str | Path = AL28_CSV, min_votes: int = 20) -> tuple[Paintings, np.ndarray]:
    """Per painting, the 9-class histogram of its non-English AL-28 votes ('other' merged into 'something else');
    paintings with fewer than min_votes such votes are dropped. Sorted by painting name."""
    frame = pd.read_csv(csv, usecols=["painting", "emotion", "language", "image_name"])
    frame = frame[frame.language.str.lower() != "english"]
    frame = frame.assign(emotion=frame.emotion.replace(AL28_MERGE))
    unknown = set(frame.emotion) - set(EMOTION_INDEX)
    if unknown:
        raise ValueError(f"unknown AL-28 labels {sorted(unknown)}")
    items = [{"painting": p, "emotion": e, "image": i} for p, e, i in zip(frame.painting, frame.emotion, frame.image_name)]
    paintings, counts = _histograms(items, lambda _: True)
    keep = counts.sum(1) >= min_votes
    return Paintings([n for n, k in zip(paintings.names, keep) if k], [i for i, k in zip(paintings.images, keep) if k]), counts[keep]


def english_counts(names: list[str], annotations_dir: str | Path) -> np.ndarray:
    """The ArtELingo English label counts of each named painting, from the train, val and test files (zeros when
    a painting has none)."""
    wanted = set(names)
    totals = {n: np.zeros(9, dtype=np.int64) for n in names}
    for split in ("train", "val", "test"):
        for item in read_json(annotations_dir, SPLIT_FILES[split]):
            if item["painting"] in wanted:
                totals[item["painting"]][EMOTION_INDEX[item["emotion"]]] += 1
    return np.stack([totals[n] for n in names])


def val_labels(annotations_dir: str | Path, heldout: frozenset[str]) -> tuple[Paintings, np.ndarray, np.ndarray]:
    """Validation paintings (held-out ones excluded) and every individual label with its painting's index."""
    items = [it for it in read_json(annotations_dir, SPLIT_FILES["val"]) if it["painting"] not in heldout]
    paintings, _ = _histograms(items, lambda _: True)
    position = {n: i for i, n in enumerate(paintings.names)}
    index = np.array([position[it["painting"]] for it in items], dtype=np.int64)
    labels = np.array([EMOTION_INDEX[it["emotion"]] for it in items], dtype=np.int64)
    return paintings, index, labels


def train_histograms(annotations_dir: str | Path, heldout: frozenset[str]) -> tuple[Paintings, np.ndarray]:
    return _histograms(read_json(annotations_dir, SPLIT_FILES["train"]), lambda p: p not in heldout)


def prior(annotations_dir: str | Path, heldout: frozenset[str]) -> np.ndarray:
    _, counts = train_histograms(annotations_dir, heldout)
    total = counts.sum(0).astype(np.float64)
    return total / total.sum()


def length_grid(annotations_dir: str | Path, tokenizer, heldout: frozenset[str], max_text_len: int) -> list[int]:
    """Caption lengths (real, non-special tokens) at the 10th, ..., 90th percentiles of ArtELingo train, truncated
    as the collator truncates (max_text_len - 2)."""
    captions = [it["caption"] for it in read_json(annotations_dir, SPLIT_FILES["train"]) if it["painting"] not in heldout]
    ids = tokenizer(captions, add_special_tokens=False)["input_ids"]
    lengths = np.minimum([len(x) for x in ids], max_text_len - 2)
    grid = np.percentile(lengths, np.arange(10, 100, 10), method="nearest")
    return [max(1, int(x)) for x in grid]


def test_captions(annotations_dir: str | Path) -> list[dict]:
    """Every caption of artelingo_test.json with its annotator's emotion index (D7)."""
    return [{"painting": it["painting"], "image": it["image"], "caption": it["caption"],
             "emotion": EMOTION_INDEX[it["emotion"]]} for it in read_json(annotations_dir, SPLIT_FILES["test"])]
```

- [ ] **Step 5: Run** `python -m pytest tests/test_hb_data.py -q`; then on the real files (CPU, a minute): `python -c "from mmae.engine.hb import data; p, c = data.al28_targets(); print(len(p.names), int(c.sum(1).min()))"`. Expected: `1600` and a minimum of at least 20.

- [ ] **Step 6: Commit** `hb: readout data (AL-28 targets, English counts, val labels, length grid)`.

---

### Task 4: GPU encode stage (`mmae/engine/hb/encode.py`, `scripts/hb_encode.py`)

**Files:**
- Create: `mmae/engine/hb/encode.py`, `scripts/hb_encode.py`, `scripts/run_hb_encode.sh`, `tests/test_hb_encode.py`

**Interfaces:**
- Consumes: `MultiMAE.decode_text` (Task 1), `mmae.engine.hb.data` (Task 3), `mmae.data.artelingo.load_painting`, `mmae.data.transforms.build_image_transform`, `mmae.models.masking.random_patch_mask`, `mmae.data.stopwords.content_token_table`, `mmae.data.artelingo.EMOTIONS`.
- Produces:
  - `load_run(run_dir, device) -> tuple[MultiMAE, DictConfig]` (builds from `config.yaml`, loads `checkpoints/best.pt`, eval mode).
  - `view_ids(n_items, n_views, seed=1234) -> Tensor (n_views, n_items, 13)` (random 25% views; `random_patch_mask(n_items, 49, 0.75, generator=torch.Generator().manual_seed(seed + v))`).
  - `hidden_captions(tokenizer, lengths, max_text_len) -> dict` of (L, T) tensors `input_ids`, `attention_mask`, `token_mask` (every real non-special token hidden).
  - `encode_paintings(model, image_paths, images_dir, transform, lengths_batch, views, batch_size, device) -> dict[str, ndarray]` with keys `dec_views` (N, V, L, 9) float16 (only for decoder models), `dec_full` (N, 1, L, 9), `emb_full` (N, D) float16, `emb_views` (N, V, D) float16.
  - `encode_d7(model, captions, images_dir, transform, tokenizer, max_text_len, n_views, batch_size, device) -> dict` with `d7_real` (n, 9 patterns, V', 9) float16, `d7_null` (n, 9, 9) float16, `d7_valid` (n, 9) bool, `d7_label` (n,) int64; patterns in the order `["j0", "prefix1", "prefix2", "prefix4", "prefix8", "random1", "random2", "random4", "random8"]`; V' = n_views for `mlm_image_source=masked`, 1 (full image) for `clean`.
  - `scripts/hb_encode.py --runs <run_dir>... --out <dir> [--images-dir --annotations-dir --al28-csv --batch-size --views 16 --d7-views 4 --skip-d7 --skip-train]` writing `<out>/<run_folder_name>/encode.npz` plus `meta.json` (arm config summary, lengths, painting name lists, seed).

- [ ] **Step 1: Failing tests** `tests/test_hb_encode.py` (CPU, tiny CLIP, fake ArtELingo). Build tiny models in-process rather than from run folders, save one with `torch.save({"model": model.state_dict()}, run/"checkpoints"/"best.pt")` plus `OmegaConf.save(cfg, run/"config.yaml")` to test `load_run`:

```python
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from helpers import compose_cfg
from mmae.data.transforms import build_image_transform
from mmae.engine.hb import encode
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP

ML80 = ("model=fusion_multilearner", "model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0")
PARCAP = ("model=fusion_multilearner", "model.emotion_head=true", "model.masking.text_ratio=1.0",
          "model.mlm_image_source=clean", "model.loss.weights.mae=0")


def make_run(tmp_path, name, *overrides):
    torch.manual_seed(0)
    cfg = compose_cfg(*overrides, f"model.backbone.pretrained={TINY_CLIP}", "data=artelingo")
    model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    OmegaConf.save(cfg, run / "config.yaml")
    torch.save({"model": model.state_dict()}, run / "checkpoints" / "best.pt")
    return run


def test_view_ids_are_fixed_and_distinct():
    a, b = encode.view_ids(5, 3), encode.view_ids(5, 3)
    assert a.shape == (3, 5, 13) and torch.equal(a, b) and not torch.equal(a[0], a[1])


def test_hidden_captions_hide_every_real_token(tokenizer):
    batch = encode.hidden_captions(tokenizer, [1, 5, 38], 40)
    assert batch["token_mask"].sum(1).tolist() == [1, 5, 38]
    assert batch["attention_mask"].sum(1).tolist() == [3, 7, 40]


@pytest.mark.parametrize("arm", ["ml80", "parcap", "contrastive"])
def test_encode_paintings_shapes(arm, tmp_path, fake_artelingo, tokenizer):
    images_dir, _, _ = fake_artelingo
    overrides = {"ml80": ML80, "parcap": PARCAP, "contrastive": ("model=contrastive",)}[arm]
    model, _ = encode.load_run(make_run(tmp_path, arm, *overrides), "cpu")
    paths = ["Style_A/p0.jpg", "Style_A/p1.jpg", "Style_A/v0.jpg"]
    out = encode.encode_paintings(model, paths, images_dir, build_image_transform("openai/clip-vit-base-patch32"),
                                  encode.hidden_captions(tokenizer, [2, 4], 40), encode.view_ids(3, 4), 2, "cpu")
    assert out["emb_full"].shape[0] == 3 and out["emb_views"].shape[:2] == (3, 4)
    if arm == "contrastive":
        assert "dec_views" not in out
    else:
        assert out["dec_views"].shape == (3, 4, 2, 9) and out["dec_full"].shape == (3, 1, 2, 9)
        assert np.isfinite(out["dec_views"].astype(np.float32)).all()


def test_encode_d7(tmp_path, fake_artelingo, tokenizer):
    images_dir, annotations, _ = fake_artelingo
    from mmae.engine.hb.data import test_captions
    model, _ = encode.load_run(make_run(tmp_path, "ml80", *ML80), "cpu")
    caps = test_captions(annotations)[:4]
    out = encode.encode_d7(model, caps, images_dir, build_image_transform("openai/clip-vit-base-patch32"),
                           tokenizer, 40, 2, 4, "cpu")
    assert out["d7_real"].shape == (4, 9, 2, 9) and out["d7_null"].shape == (4, 9, 9)
    assert out["d7_valid"][:, 0].all() and out["d7_label"].shape == (4,)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_hb_encode.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement** `mmae/engine/hb/encode.py`:

```python
"""GPU stage of the H-b readouts (spec sections 6, 8, 9): forward passes only, saved as arrays for the CPU stage.

For each painting: the emotion logits of the caption decoder with the caption fully hidden, for every caption length
of the grid, on V random 25% views and on the full image (decoder models only); the pooled image embedding of the
full image and of each view (every model). For D7: the emotion logits for every test caption with j content tokens
visible (prefix and random subset), on the real image (views or full) and on the null image."""
from __future__ import annotations

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
    return model.reconstruction and model.use_text and model.emotion_head


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
    return DataLoader(_Images(paths, images_dir, transform), batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def _emotion_logits(model, images, hidden, ids_keep) -> torch.Tensor:
    """(B, L, 9): every caption length of `hidden` for every image, one decoder pass per length."""
    out = []
    for l in range(hidden["input_ids"].shape[0]):
        rep = {k: v[l : l + 1].expand(images.shape[0], -1) for k, v in hidden.items()}
        _, emotion = model.decode_text(images, rep["input_ids"], rep["attention_mask"], rep["token_mask"], ids_keep)
        out.append(emotion.float())
    return torch.stack(out, dim=1)


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
        chunks["emb_full"].append(model.vision.pool(model.vision.encode(images)).half().cpu())
        chunks["emb_views"].append(torch.stack(
            [model.vision.pool(model.vision.encode(images, ids)) for ids in batch_views], dim=1).half().cpu())
        if decoder:
            chunks["dec_views"].append(torch.stack(
                [_emotion_logits(model, images, hidden, ids) for ids in batch_views], dim=1).half().cpu())
            chunks["dec_full"].append(_emotion_logits(model, images, hidden, None).unsqueeze(1).half().cpu())
    return {k: torch.cat(v).numpy() for k, v in chunks.items()}


@torch.no_grad()
def prompt_embeddings(model, tokenizer, max_text_len, device, labels) -> np.ndarray:
    enc = tokenizer([f"a painting that evokes {label}." for label in labels], max_length=max_text_len,
                    truncation=True, padding="max_length", return_tensors="pt")
    return model.embed_text(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu().numpy()


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
        per_pattern_real, per_pattern_null = [], []
        for k in range(p):
            tm = token_masks[index, k].to(device)
            if masked_source:
                per_view = [model.decode_text(images, ids, am, tm, views[v, index].to(device))[1].float()
                            for v in range(n_views)]
            else:
                per_view = [model.decode_text(images, ids, am, tm, None)[1].float()]
            per_pattern_real.append(torch.stack(per_view, dim=1))
            per_pattern_null.append(model.decode_text(torch.zeros_like(images), ids, am, tm, None)[1].float())
        real_out.append(torch.stack(per_pattern_real, dim=1).half().cpu())
        null_out.append(torch.stack(per_pattern_null, dim=1).half().cpu())
    return {"d7_real": torch.cat(real_out).numpy(), "d7_null": torch.cat(null_out).numpy(),
            "d7_valid": valid.numpy(), "d7_label": np.array([c["emotion"] for c in captions], dtype=np.int64)}
```

Note: the null image for masked-source models also uses all patches (`ids_keep=None`): with no image content, the view is irrelevant; this is stated in the report.

`scripts/hb_encode.py` (argparse; for each run dir: `load_run`; `length_grid` once per data root (cache it as `<out>/length_grid.json`); AL-28 target paintings (`al28_targets`), validation paintings (`val_labels`), train paintings (`train_histograms`, embeddings of the full image only via `encode_paintings`-style pooled pass; skip with `--skip-train`); `prompt_embeddings` with `EMOTIONS`; `logit_scale` (`model.logit_scale.exp().item()` if present); D7 unless `--skip-d7` or the model has no decoder; save `np.savez_compressed(out/run_name/"encode.npz", al28_*=..., val_*=..., train_emb=..., prompts=..., logit_scale=..., **d7)` and `meta.json` with `{"run": str(run_dir), "arm": cfg.wandb.name or cfg.model.name, "seed": cfg.seed, "mlm_image_source": ..., "text_ratio": ..., "mae_weight": ..., "lengths": grid, "al28": names, "val": names, "train": names}`. Use `torch.autocast("cuda", dtype=torch.bfloat16)` on CUDA for the forward passes; logits are cast to float before saving. Log progress per stage with timings.

`scripts/run_hb_encode.sh` (cluster launch form; mirrors `scripts/run_diagnostics.sh`):

```bash
#!/usr/bin/env bash
# H-b GPU readout stage on a node: bash scripts/run_hb_encode.sh --runs <dir>... --out <dir> [hb_encode.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/hb_encode.py "$@"
```

- [ ] **Step 4: Run** `python -m pytest tests/test_hb_encode.py -q`. Expected: PASS.

- [ ] **Step 5: Script smoke on CPU** with a fake run folder from the test helper (`make_run`) and the fake ArtELingo plus a fake AL-28 CSV (`make_fake_al28`): `python scripts/hb_encode.py --runs <run> --out <tmp> --images-dir <...> --annotations-dir <...> --al28-csv <...> --batch-size 2 --views 2 --d7-views 2` exits 0 and writes `encode.npz` and `meta.json`. Add this as a test in `tests/test_hb_encode.py` (subprocess, `CUDA_VISIBLE_DEVICES=""`).

- [ ] **Step 6: Commit** `hb: GPU encode stage (emotion logits per view and length, embeddings, D7)`.

---

### Task 5: Baselines (`mmae/engine/hb/baselines.py`)

**Files:**
- Create: `mmae/engine/hb/baselines.py`, `tests/test_hb_baselines.py`

**Interfaces:**
- Produces:
  - `prompt_logits(emb (N, D) or (N, V, D), prompts (9, D), logit_scale: float) -> ndarray (N, M, 9)` (M = 1 or V), embeddings L2-normalised first.
  - `@dataclass SoftProbe: weight (D, 9); bias (9,); weight_decay: float; val_nll: float` with `logits(x (N, D) or (N, V, D)) -> (N, M, 9)`.
  - `fit_soft_probe(train_x, train_counts, val_x, val_index, val_labels, weight_decays=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2), steps=200) -> SoftProbe` (full-batch L-BFGS in torch float32 on CPU; soft cross-entropy + wd * ||W||^2; picks the decay with the lowest val NLL at T = 1).

- [ ] **Step 1: Failing tests** `tests/test_hb_baselines.py`:

```python
import numpy as np

from mmae.engine.hb import baselines, calibrate

rng = np.random.default_rng(0)


def test_prompt_logits_shapes_and_scale():
    emb, prompts = rng.normal(size=(4, 8)), rng.normal(size=(9, 8))
    out = baselines.prompt_logits(emb, prompts, 100.0)
    assert out.shape == (4, 1, 9) and np.abs(out).max() <= 100.0 + 1e-6
    assert baselines.prompt_logits(rng.normal(size=(4, 3, 8)), prompts, 100.0).shape == (4, 3, 9)


def test_soft_probe_learns_a_linear_rule_and_picks_a_decay():
    w = rng.normal(size=(8, 9))
    x = rng.normal(size=(600, 8))
    probs = np.exp(x @ w) / np.exp(x @ w).sum(1, keepdims=True)
    counts = np.stack([rng.multinomial(5, p) for p in probs])
    val_x = rng.normal(size=(200, 8))
    val_p = np.exp(val_x @ w) / np.exp(val_x @ w).sum(1, keepdims=True)
    val_index = np.repeat(np.arange(200), 5)
    val_labels = np.array([rng.choice(9, p=val_p[i]) for i in val_index])
    probe = baselines.fit_soft_probe(x, counts, val_x, val_index, val_labels)
    assert probe.weight_decay in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    fitted = calibrate.mixture(probe.logits(val_x), 1.0)
    prior = np.tile(counts.sum(0) / counts.sum(), (200, 1))
    assert calibrate.nll(fitted, val_index, val_labels) < calibrate.nll(prior, val_index, val_labels)
```

- [ ] **Step 2: Run**, expect FAIL; **Step 3: Implement**:

```python
"""Dual-encoder baselines for D4 (spec section 6.3): prompt softmax and the soft-label linear probe."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mmae.engine.hb import calibrate

WEIGHT_DECAYS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def prompt_logits(emb: np.ndarray, prompts: np.ndarray, logit_scale: float) -> np.ndarray:
    emb = _unit(emb)
    if emb.ndim == 2:
        emb = emb[:, None]
    return logit_scale * emb @ _unit(prompts).T


@dataclass
class SoftProbe:
    weight: np.ndarray
    bias: np.ndarray
    weight_decay: float
    val_nll: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = _unit(x)
        if x.ndim == 2:
            x = x[:, None]
        return x @ self.weight + self.bias


def _fit(x: torch.Tensor, targets: torch.Tensor, weight_decay: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    weight = torch.zeros(x.shape[1], targets.shape[1], requires_grad=True)
    bias = torch.zeros(targets.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        log_probs = torch.log_softmax(x @ weight + bias, dim=1)
        loss = -(targets * log_probs).sum(1).mean() + weight_decay * (weight**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach().numpy().astype(np.float64), bias.detach().numpy().astype(np.float64)


def fit_soft_probe(train_x, train_counts, val_x, val_index, val_labels, weight_decays=WEIGHT_DECAYS,
                   steps: int = 200) -> SoftProbe:
    x = torch.tensor(_unit(train_x), dtype=torch.float32)
    counts = np.asarray(train_counts, np.float64)
    targets = torch.tensor(counts / counts.sum(1, keepdims=True), dtype=torch.float32)
    best = None
    for wd in weight_decays:
        weight, bias = _fit(x, targets, wd, steps)
        probe = SoftProbe(weight, bias, wd, float("nan"))
        probe.val_nll = calibrate.nll(calibrate.mixture(probe.logits(val_x), 1.0), val_index, val_labels)
        if best is None or probe.val_nll < best.val_nll:
            best = probe
    return best
```

- [ ] **Step 4: Run** tests, PASS; **Step 5: Commit** `hb: prompt-softmax and soft-label probe baselines`.

---

### Task 6: Hierarchical bootstrap and the decision rule (`mmae/engine/hb/bootstrap.py`)

**Files:**
- Create: `mmae/engine/hb/bootstrap.py`, `tests/test_hb_bootstrap.py`

**Interfaces:**
- Produces:
  - `holm(pvalues: list[float]) -> list[float]` (same as `tests/20261003_ml_improve/stage2_table.py`).
  - `two_sided_p(diffs: ndarray) -> float` = `min(1, 2 * min(mean(diffs <= 0), mean(diffs >= 0)))`.
  - `paired_bootstrap(a: list[ndarray (N, 9)], b: list[ndarray (N, 9)], human (N, 9), metric: str, B=10000, seed=0) -> dict(diff, ci_low, ci_high, p, draws)` where `a`, `b` are per-seed predicted distributions of two arms, `metric` in `{"jsd", "entropy_spearman"}`; diff = statistic(a) - statistic(b) with statistic = mean over the drawn seeds of the metric on the drawn paintings; seeds resampled with replacement independently per arm; one painting resample per replicate shared by both arms.
  - `decide(results: dict[(metric, comparator)] -> dict with "p") -> dict` applying Holm over the four tests and the support rule; JSD "favours ML-80" when diff < 0 (ML-80 minus comparator), entropy Spearman when diff > 0.

- [ ] **Step 1: Failing tests** `tests/test_hb_bootstrap.py`:

```python
import numpy as np
import pytest

from mmae.engine.hb import bootstrap

rng = np.random.default_rng(1)
H = rng.dirichlet(np.ones(9), size=300)


def arm(scale, seeds=3):
    """Per-seed distributions: the human target plus noise of the given scale, renormalised."""
    out = []
    for _ in range(seeds):
        x = np.abs(H + rng.normal(scale=scale, size=H.shape)) + 1e-9
        out.append(x / x.sum(1, keepdims=True))
    return out


def test_holm_matches_reference():
    assert bootstrap.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_detects_a_clear_difference_and_not_a_null():
    good, bad = arm(0.01), arm(0.2)
    clear = bootstrap.paired_bootstrap(good, bad, H, "jsd", B=2000)
    assert clear["diff"] < 0 and clear["p"] < 0.01
    same = bootstrap.paired_bootstrap(good, arm(0.01), H, "jsd", B=2000)
    assert same["p"] > 0.05


def test_paintings_are_paired_across_arms():
    """Two identical one-seed arms differ by exactly zero in every replicate, which holds only if both arms are
    scored on the same painting sample (seed resampling cannot differ with one seed per arm)."""
    one = arm(0.05, seeds=1)
    for metric in ("jsd", "entropy_spearman"):
        single = bootstrap.paired_bootstrap(one, [one[0].copy()], H, metric, B=500)
        assert np.all(single["draws"] == 0.0)


def test_decide_support_rule():
    p = {("jsd", "parcap"): 0.001, ("jsd", "probe"): 0.002, ("entropy_spearman", "parcap"): 0.5,
         ("entropy_spearman", "probe"): 0.6}
    diffs = {("jsd", "parcap"): -0.01, ("jsd", "probe"): -0.02, ("entropy_spearman", "parcap"): 0.01,
             ("entropy_spearman", "probe"): -0.01}
    out = bootstrap.decide({k: {"p": p[k], "diff": diffs[k]} for k in p})
    assert out["support"] is True and out["supporting_metric"] == "jsd"
    flipped = {k: {"p": p[k], "diff": -diffs[k]} for k in p}
    assert bootstrap.decide(flipped)["support"] is False
```

- [ ] **Step 2: Run**, expect FAIL; **Step 3: Implement**:

```python
"""Hierarchical bootstrap and the pre-registered D4 decision rule (spec section 7)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from mmae.engine.hb import metrics

COMPARATORS = ("parcap", "probe")
METRICS = ("jsd", "entropy_spearman")
BETTER = {"jsd": -1, "entropy_spearman": +1}  # sign of (ML-80 - comparator) that favours ML-80


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def two_sided_p(diffs: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def paired_bootstrap(a, b, human, metric: str, B: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = human.shape[0]
    if metric == "jsd":
        per_a = [metrics.js_distance(x, human) for x in a]
        per_b = [metrics.js_distance(x, human) for x in b]

        def stat(per, seeds, idx):
            return float(np.mean([per[s][idx].mean() for s in seeds]))
    elif metric == "entropy_spearman":
        human_entropy = metrics.entropy_bits(human)
        per_a = [metrics.entropy_bits(x) for x in a]
        per_b = [metrics.entropy_bits(x) for x in b]

        def stat(per, seeds, idx):
            return float(np.mean([_spearman_rows(per[s][idx], human_entropy[idx]) for s in seeds]))
    else:
        raise ValueError(metric)
    full = np.arange(n)
    diff = stat(per_a, range(len(a)), full) - stat(per_b, range(len(b)), full)
    draws = np.empty(B)
    for r in range(B):
        idx = rng.integers(0, n, n)
        sa = rng.integers(0, len(a), len(a))
        sb = rng.integers(0, len(b), len(b))
        draws[r] = stat(per_a, sa, idx) - stat(per_b, sb, idx)
    return {"diff": diff, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p": two_sided_p(draws), "draws": draws}


def decide(results: dict) -> dict:
    keys = [(m, c) for m in METRICS for c in COMPARATORS]
    adjusted = dict(zip(keys, holm([results[k]["p"] for k in keys])))
    favours = {k: np.sign(results[k]["diff"]) == BETTER[k[0]] and adjusted[k] < 0.05 for k in keys}
    supporting = [m for m in METRICS if all(favours[(m, c)] for c in COMPARATORS)]
    return {"holm": {f"{m}/{c}": adjusted[(m, c)] for m, c in keys},
            "favours": {f"{m}/{c}": bool(favours[(m, c)]) for m, c in keys},
            "support": bool(supporting), "supporting_metric": supporting[0] if supporting else None}
```

- [ ] **Step 4: Run** tests, PASS (B = 10,000 at 1,600 paintings and 3 seeds runs in about a minute; keep tests at B <= 2,000). **Step 5: Commit** `hb: hierarchical bootstrap and the D4 decision rule`.

---

### Task 7: CPU stage: gate check, D4 table, D6, D7 (`mmae/engine/hb/analysis.py`, `scripts/hb_gate.py`, `scripts/hb_table.py`)

**Files:**
- Create: `mmae/engine/hb/analysis.py`, `scripts/hb_gate.py`, `scripts/hb_table.py`, `tests/test_hb_analysis.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 5, 6 and the `encode.npz` / `meta.json` layout of Task 4.
- Produces:
  - `load_encoded(run_out_dir) -> dict` (arrays + meta).
  - `readouts(enc) -> dict[str, (al28_logits (N, M, 9), val_logits (P, M, 9))]` for the decoder arms: `"views16"` (dec_views reshaped to M = V x L), `"views4"`, `"views1"` (first 4 / 1 views), `"full"` (dec_full); for every arm: `"prompt_full"`, `"prompt_views16"` (from embeddings and prompts).
  - `gate(enc, prior, val_index, val_labels) -> dict(passed: bool, temperature, val_nll, prior_nll, jsd_to_prior, checks: dict[str, bool])` using the in-distribution readout (`views16` when `mlm_image_source == "masked"`, else `full`).
  - `scripts/hb_gate.py --encoded <dir>... [--annotations-dir]` prints one line per run and exits 1 if any run fails a check.
  - `scripts/hb_table.py --encoded-root <dir> --out <dir> [--al28-csv --annotations-dir --B 10000]` writes `<out>/hb_d4.json` and `<out>/hb_d4.md`: per arm and seed the temperatures and every metric (primary and secondary, 8-named version, "other dropped" version), the probe per arm and seed (chosen decay, val NLL) and the strongest probe per seed, prompt softmax, prior-only, the split-half ceiling and the English reference, the four bootstrap tests with Holm and the support verdict, the D6 table (K in {1, 4, 16}, full image, between-view mutual information Spearman with a bootstrap interval, Par-cap on views, C's prompt softmax and the strongest probe on views), and the D7 table (log-loss and accuracy of the annotator's emotion per pattern, real vs null image with a bootstrap interval of the difference over captions and seeds; counts of captions per pattern).

For the "other dropped" version, `al28_targets` gains `drop_other: bool = False` (votes labelled "other" removed instead of merged); for the 8-named version, renormalise model and human over the first 8 classes.

Split-half ceiling: for each painting, 10 random splits of its non-English votes into two halves (seeded); JSD between halves (mean over splits) and entropy Spearman between halves; this needs the raw vote list, so `al28_targets` also returns (or a sibling `al28_votes(csv)` returns) per-painting arrays of vote labels.

- [ ] **Step 1: Failing tests** `tests/test_hb_analysis.py`: build synthetic `encode.npz` + `meta.json` for three fake runs (an ML-80-like masked decoder arm with dec_views/dec_full, a Par-cap-like clean arm, a contrastive arm without decoder arrays) with small N (al28 30 paintings, val 40 paintings, train 50) and random arrays; a fake AL-28 CSV covering the 30 paintings; then assert:
  - `gate` passes for logits whose softmax tracks the val labels and fails (`checks["jsd_to_prior"] is False`) when every logit row equals log(prior);
  - `scripts/hb_table.py` on the folder (subprocess, `--B 200`) exits 0 and writes `hb_d4.json` with keys `arms`, `tests`, `decision`, `d6`, `d7`, `references`;
  - the contrastive arm has prompt and probe entries but no decoder entries.

- [ ] **Step 2: Run**, expect FAIL; **Step 3: Implement** `analysis.py` with the functions above (reuse `calibrate.mixture`, `fit_temperature`, `nll`; `metrics.*`; `baselines.*`; `bootstrap.paired_bootstrap`, `decide`). Arms are identified from `meta.json`: Par-cap = `text_ratio == 1.0 and mlm_image_source == "clean"`; ML-80 = `text_ratio == 0.8 and mae_weight == 0`; ML-80+MAE = `text_ratio == 0.8 and mae_weight > 0`; C = no decoder. The ML-80 primary readout is `views16`, Par-cap's is `full`. Probes: fit per (arm, seed) on `train_emb` with `train_histograms` counts; temperatures are then fitted for the probe like any readout.

- [ ] **Step 4: Run** tests, PASS. **Step 5: Commit** `hb: gate check and the D4, D6, D7 tables`.

---

### Task 8 (controller): run the gate on wave 1, then the full readouts

- [ ] Pull the first finished wave-1 runs (`cluster pull --tag <tag>`), check the local GPU (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`), and run `flock -n -o -E 75 /tmp/gpu0.lock python scripts/hb_encode.py --runs <ML-80 s42> <Par-cap s42> <ML-80+MAE s42> --out res/hb/d4 --skip-d7` then `python scripts/hb_gate.py --encoded res/hb/d4/*`. If the local GPU is held, run `bash scripts/run_hb_encode.sh` through `cluster launch` on the node holding the runs instead.
- [ ] Gate passes: launch wave 2. Gate fails: stop, debug (superpowers:systematic-debugging), no wave 2.
- [ ] When all 12 runs are in: `hb_encode.py` over all of them (with D7), then `hb_table.py --encoded-root res/hb/d4 --out res/hb/d4_tables`; log the verdict in `tests/20261007_hb/runs.md`.
