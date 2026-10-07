# H-b Plan 1: ArtELingo training path (emotion head, text ratio 1.0, loader, configs) and the D4 launch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the four D4 arms (C, ML-80, ML-80+MAE, Par-cap) on ArtELingo English with an always-hidden 9-way emotion slot in the caption decoder, and launch them on DAS6.

**Architecture:** A 9-way emotion head rides on one extra learned query of the existing text `QueryDecoder`; its cross-entropy joins the MLM mean. `random_token_mask` accepts ratio 1.0 (Par-cap). A new `mmae/data/artelingo.py` provides pairs (image, caption, emotion) and 5-caption retrieval sets with the AL-28 paintings held out of train and validation; a small factory picks COCO or ArtELingo from `data.name`. Plans 2 (D4/D6/D7 readouts) and 3 (D1/D2 COCO probes) follow.

**Tech Stack:** PyTorch 2.11, HF transformers CLIP, Hydra, Accelerate, pytest. Python: `/root/miniconda3/envs/MultiMAE/bin/python`.

**Spec:** `docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md` (sections 4, 5, 12, 13).

## Global Constraints

- Emotion classes, in this order everywhere: `("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else")`.
- The emotion is always a target, never an input: no token, no text-tower position.
- `loss_mlm = (sum of masked-token CE + sum of emotion CE) / (number of masked tokens + number of captions)`; `loss_mlm_tokens` and `loss_emotion` are logged, not added to `loss`.
- `model.emotion_head` defaults to false and is read with `cfg.get` (old run configs build).
- Text ratio range is (0, 1]; at 1.0 every real non-special token is masked, BOS/EOS/padding never.
- AL-28 paintings (1,658, `mmae/data/al28_paintings.txt`) never reach train or validation data; test keeps them.
- `data.max_text_len = 40` for ArtELingo; COCO configs are not edited (code treats a missing `data.name` as `coco`).
- Extended COCO metrics (ECCV, CxC, PMRP) never run on ArtELingo.
- Fast tests run on CPU with the tiny CLIP: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/MultiMAE/bin/python -m pytest -q -x -n 8` (pytest-xdist is installed if `-n` works; otherwise drop `-n 8`).
- Commits: `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit ...` (the container has no global git identity), message ending with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01ApMQ4krinxjrUg4cqirWE2`.
- Edits to existing source files get an entry in `.claude/20261007_log.md` (one `# <path>` header per file, before/after snippet, why).

## Review Focus

1. Old COCO run configs (no `emotion_head`, no `data.name`) must still build and evaluate unchanged: Task 2 and Task 4 tests pin it.
2. A batch without `emotion` reaching a model with `emotion_head` must fail loudly, not train silently without the emotion loss: Task 2 test.
3. Held-out paintings must not leak into train or val through either the pairs or the retrieval path: Task 3 tests both.
4. At text ratio 1.0 the caption decoder must not see any caption content (only length): Task 1 test.
5. The emotion slot must read visible caption tokens (D7 needs it) but never `batch["emotion"]`: Task 2 test.

---

### Task 1: Text masking ratio 1.0 (Par-cap)

**Files:**
- Modify: `mmae/models/masking.py` (`random_token_mask`)
- Modify: `tests/test_model_variants.py` (VARIANTS, leak test), `tests/test_masking.py` (create the file if it does not exist)

**Interfaces:**
- Produces: `random_token_mask(attention_mask, special_tokens_mask, ratio, generator=None, allowed=None)` accepting `0 < ratio <= 1`.

- [ ] **Step 1: Write the failing tests** in `tests/test_masking.py`:

```python
import pytest
import torch

from helpers import make_batch
from mmae.models.masking import random_token_mask


def test_ratio_one_masks_every_real_token_and_nothing_else(tokenizer):
    batch = make_batch(tokenizer)
    mask = random_token_mask(batch["attention_mask"], batch["special_tokens_mask"], 1.0)
    real = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    assert torch.equal(mask, real)


@pytest.mark.parametrize("ratio", [0.0, 1.01, -0.5])
def test_ratio_outside_zero_one_is_rejected(tokenizer, ratio):
    batch = make_batch(tokenizer)
    with pytest.raises(ValueError):
        random_token_mask(batch["attention_mask"], batch["special_tokens_mask"], ratio)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_masking.py -q`. Expected: the ratio-1.0 test FAILS with `ValueError: token mask ratio must be in (0, 1)`.

- [ ] **Step 3: Implement.** In `random_token_mask` replace the check and the docstring's range:

```python
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"token mask ratio must be in (0, 1], got {ratio}")
```

(At 1.0, `k = round(n)` = n and `min(k, available)` = n, so every real non-special token is masked.)

- [ ] **Step 4: Add the Par-cap model variant.** In `tests/test_model_variants.py` add to `VARIANTS`:

```python
    "parcap": ("model.masking.text_ratio=1.0", "model.mlm_image_source=clean", "model.loss.weights.mae=0"),
```

and add `NO_VISIBLE_TEXT = {"parcap"}` below `VARIANTS`. In `test_each_decoder_ignores_its_own_masked_content`, guard the last assertion (visible text moves the text decoder), since a 100% mask leaves no visible text:

```python
    if variant not in NO_VISIBLE_TEXT:
        assert not torch.equal(run(change(batch, model, masks, "text", False))[0]["text_decoder"], base["text_decoder"])
```

Then add the dedicated test:

```python
def test_parcap_text_decoder_sees_no_caption_content(tokenizer, monkeypatch):
    """At text ratio 1.0 the caption decoder's output depends on the image and the caption length only."""
    model = tiny_model("fusion_multilearner", *VARIANTS["parcap"]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[0]["text_decoder"]
    real = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    assert torch.equal(masks["token"], real)
    other = dict(batch, input_ids=torch.where(real, torch.randint(1000, 40000, batch["input_ids"].shape), batch["input_ids"]))
    assert torch.equal(run(other)[0]["text_decoder"], base)
    noisy = dict(batch, pixel_values=batch["pixel_values"] + torch.randn_like(batch["pixel_values"]))
    assert not torch.equal(run(noisy)[0]["text_decoder"], base)
```

- [ ] **Step 5: Confirm the new guard bites.** Temporarily change `random_token_mask` to mask `n - 1` tokens at ratio 1.0 (e.g. `k = torch.clamp(k - (ratio == 1.0), min=1)`), run `python -m pytest tests/test_masking.py tests/test_model_variants.py -q -k "ratio_one or parcap"`, see both fail, then revert.

- [ ] **Step 6: Run** `python -m pytest tests/test_masking.py tests/test_model_variants.py tests/test_model.py -q`. Expected: all PASS.

- [ ] **Step 7: Change log and commit.** Append to `.claude/20261007_log.md` the `mmae/models/masking.py` entry; commit `masking: allow text ratio 1.0 (Par-cap, H-b spec 2026-10-07)`.

---

### Task 2: Emotion head on the caption decoder

**Files:**
- Modify: `mmae/models/decoders.py` (`QueryDecoder`), `mmae/models/model.py` (`MultiMAE.__init__`, `forward`), `mmae/losses.py` (new `mlm_emotion_loss`), `configs/model/base.yaml` (new key)
- Modify: `tests/test_model.py` (`recorder` hook), `tests/helpers.py` (`add_emotion`), `tests/test_model_variants.py` (VARIANTS, `variant_batch`)
- Create: `tests/test_emotion_head.py`

**Interfaces:**
- Consumes: `random_token_mask` (Task 1).
- Produces:
  - `QueryDecoder(num_queries, dim, out_dim, depth=4, heads=8, dropout=0.1, prefix_queries=0, prefix_out_dim=0)`; with `prefix_queries > 0`, `forward(memory, memory_padding=None, query_padding=None)` returns `(main (B, n, out_dim), prefix (B, prefix_queries, prefix_out_dim))`, else the main tensor only. Parameter names: `prefix_queries`, `prefix_head`.
  - `mmae.models.model.NUM_EMOTIONS = 9`; `MultiMAE.emotion_head: bool`.
  - `mmae.losses.mlm_emotion_loss(logits, input_ids, token_mask, emotion_logits, emotion) -> tuple[Tensor, dict[str, Tensor]]`.
  - `MultiMAE.forward(batch)` with `emotion_head` needs `batch["emotion"]` (B,) long and returns `loss_mlm`, `loss_mlm_tokens`, `loss_emotion` among its keys.
  - `tests/helpers.add_emotion(batch) -> dict`.

- [ ] **Step 1: Test helpers.** In `tests/helpers.py` add:

```python
def add_emotion(batch: dict) -> dict:
    """An ArtELingo-style emotion label per caption (class index in 0..8)."""
    b = batch["input_ids"].shape[0]
    return {**batch, "emotion": torch.arange(b) % 9}
```

In `tests/test_model.py::recorder`, replace the decoder hook so a decoder returning `(main, prefix)` is captured as `captured[dec]` (main) and `captured[dec + "_prefix"]` (prefix):

```python
    def capture(key):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().clone()
                captured[key + "_prefix"] = output[1].detach().clone()
            else:
                captured[key] = output.detach().clone()
        return hook

    for dec in ("image_decoder", "text_decoder"):
        if hasattr(model, dec):
            getattr(model, dec).register_forward_hook(capture(dec))
```

- [ ] **Step 2: Write the failing tests** in `tests/test_emotion_head.py`:

```python
"""H-b emotion head (spec 2026-10-07, section 5.1): one extra text-decoder query, 9 classes, always a target."""
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from helpers import add_emotion, compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP
from mmae.models.decoders import QueryDecoder
from mmae.models.model import NUM_EMOTIONS
from test_model import recorder, tiny_model

ML80 = ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0")


def test_query_decoder_prefix_shapes_and_backward_compatibility():
    torch.manual_seed(0)
    memory = torch.randn(2, 5, 16)
    plain = QueryDecoder(7, 16, 11, depth=1, heads=2)
    assert plain(memory).shape == (2, 7, 11)
    with_prefix = QueryDecoder(7, 16, 11, depth=1, heads=2, prefix_queries=1, prefix_out_dim=9)
    padding = torch.zeros(2, 7, dtype=torch.bool)
    padding[1, 4:] = True
    main, prefix = with_prefix(memory, query_padding=padding)
    assert main.shape == (2, 7, 11) and prefix.shape == (2, 1, 9)
    assert torch.isfinite(main).all() and torch.isfinite(prefix).all()
    assert {"prefix_queries", "prefix_head.weight", "prefix_head.bias"} <= {n for n, _ in with_prefix.named_parameters()}
    assert not any(n.startswith("prefix") for n, _ in plain.named_parameters())


def test_emotion_loss_joins_the_mlm_mean(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", *ML80).eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_emotion(make_batch(tokenizer))
    captured, out, _ = run(batch)
    token_mask = masks["token"]
    logits, emotion_logits = captured["text_decoder"], captured["text_decoder_prefix"][:, 0]
    token_ce = F.cross_entropy(logits[token_mask].float(), batch["input_ids"][token_mask], reduction="sum")
    emotion_ce = F.cross_entropy(emotion_logits.float(), batch["emotion"], reduction="sum")
    n, b = token_mask.sum(), batch["emotion"].shape[0]
    torch.testing.assert_close(out["loss_mlm"], (token_ce + emotion_ce) / (n + b))
    torch.testing.assert_close(out["loss_mlm_tokens"], token_ce / n)
    torch.testing.assert_close(out["loss_emotion"], emotion_ce / b)
    weights = model.loss_weights
    expected = weights["contrastive"] * out["loss_contrastive"] + weights["mae"] * out["loss_mae"] + weights["mlm"] * out["loss_mlm"]
    torch.testing.assert_close(out["loss"], expected)


def test_emotion_is_a_target_never_an_input(tokenizer, monkeypatch):
    """Changing the labels moves the loss but no decoder output; the slot reads visible text, not hidden text."""
    model = tiny_model("fusion_multilearner", *ML80).eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_emotion(make_batch(tokenizer))
    base_out, base_loss, _ = run(batch)
    relabelled = dict(batch, emotion=(batch["emotion"] + 1) % NUM_EMOTIONS)
    out, loss, _ = run(relabelled)
    assert torch.equal(out["text_decoder_prefix"], base_out["text_decoder_prefix"])
    assert not torch.equal(loss["loss_emotion"], base_loss["loss_emotion"])
    hidden = masks["token"]
    visible = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool() & ~hidden
    noise = torch.randint(1000, 40000, batch["input_ids"].shape)
    assert torch.equal(run(dict(batch, input_ids=torch.where(hidden, noise, batch["input_ids"])))[0]["text_decoder_prefix"],
                       base_out["text_decoder_prefix"])
    assert not torch.equal(run(dict(batch, input_ids=torch.where(visible, noise, batch["input_ids"])))[0]["text_decoder_prefix"],
                           base_out["text_decoder_prefix"])


def test_emotion_head_needs_labels(tokenizer):
    model = tiny_model("fusion_multilearner", *ML80)
    with pytest.raises(KeyError, match="emotion"):
        model(make_batch(tokenizer))


@pytest.mark.parametrize("name", ["contrastive", "image_mae"])
def test_emotion_head_needs_the_text_decoder(name):
    with pytest.raises(ValueError, match="emotion_head"):
        tiny_model(name, "model.emotion_head=true")


def test_old_configs_build_without_the_key():
    cfg = compose_cfg("model=fusion_multilearner", f"model.backbone.pretrained={TINY_CLIP}")
    old = OmegaConf.to_container(cfg.model)
    old.pop("emotion_head", None)
    model = MultiMAE(OmegaConf.create(old), max_text_len=cfg.data.max_text_len)
    assert model.emotion_head is False
    assert not any("prefix" in n for n, _ in model.named_parameters())


def test_prefix_query_gets_no_weight_decay():
    model = tiny_model("fusion_multilearner", *ML80)
    prefix = model.text_decoder.prefix_queries
    groups = model.param_groups(1e-4, 1e-5, 0.05)
    owner = [g for g in groups if any(p is prefix for p in g["params"])]
    assert len(owner) == 1 and owner[0]["weight_decay"] == 0.0
```

Also extend `tests/test_model_variants.py`:

```python
    "emotion_head": ("model.emotion_head=true",),
    "emotion_ml80": ("model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0"),
```

change the Task 1 entry to `"parcap": ("model.emotion_head=true", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean", "model.loss.weights.mae=0"),`, add `EMOTION_VARIANTS = {"emotion_head", "emotion_ml80", "parcap"}`, and make `variant_batch` add labels:

```python
def variant_batch(tokenizer, variant: str) -> dict:
    batch = add_content_mask(make_batch(tokenizer), tokenizer) if variant == "m2b_content" else make_batch(tokenizer)
    return add_emotion(batch) if variant in EMOTION_VARIANTS else batch
```

(`test_parcap_text_decoder_sees_no_caption_content` from Task 1 then needs `batch = add_emotion(make_batch(tokenizer))` and `noisy`/`other` built from it.)

- [ ] **Step 3: Run** `python -m pytest tests/test_emotion_head.py tests/test_model_variants.py -q`. Expected: FAIL (`ImportError: cannot import name 'NUM_EMOTIONS'`, unexpected keyword `prefix_queries`).

- [ ] **Step 4: Implement `QueryDecoder` prefix queries** (`mmae/models/decoders.py`):

```python
class QueryDecoder(nn.Module):
    """Predicts one output vector per query; works with a memory of any length.

    With `prefix_queries` > 0, that many extra learned queries sit ahead of the positional queries (no position
    embedding, never padded) and read out through their own head of size `prefix_out_dim`; forward then returns
    (main outputs, prefix outputs). H-b's emotion slot is one such query (spec 2026-10-07, section 5.1).
    """

    def __init__(
        self, num_queries: int, dim: int, out_dim: int, depth: int = 4, heads: int = 8, dropout: float = 0.1,
        prefix_queries: int = 0, prefix_out_dim: int = 0,
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
        self.prefix_queries = None
        if prefix_queries > 0:
            if prefix_out_dim <= 0:
                raise ValueError("prefix_queries needs prefix_out_dim > 0")
            self.prefix_queries = nn.Parameter(torch.empty(prefix_queries, dim))
            nn.init.normal_(self.prefix_queries, std=0.02)
            self.prefix_head = nn.Linear(dim, prefix_out_dim)

    @property
    def num_queries(self) -> int:
        return self.queries.shape[0]

    def forward(
        self,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        query_padding: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """memory (B, L, dim); paddings are bool, True = padded. Returns (B, n, out_dim), plus
        (B, prefix_queries, prefix_out_dim) when the decoder has prefix queries."""
        n = self.num_queries if query_padding is None else query_padding.shape[1]
        if n > self.num_queries:
            raise ValueError(f"{n} queries requested but the decoder has {self.num_queries}")
        batch = memory.shape[0]
        target = (self.queries[:n] + self.pos_embed[:n]).unsqueeze(0).expand(batch, -1, -1)
        p = 0
        if self.prefix_queries is not None:
            p = self.prefix_queries.shape[0]
            target = torch.cat([self.prefix_queries.unsqueeze(0).expand(batch, -1, -1), target], dim=1)
            if query_padding is not None:
                query_padding = torch.cat([query_padding.new_zeros(batch, p), query_padding], dim=1)
        out = self.decoder(
            target, memory, tgt_key_padding_mask=query_padding, memory_key_padding_mask=memory_padding
        )
        if self.prefix_queries is None:
            return self.head(out)
        return self.head(out[:, p:]), self.prefix_head(out[:, :p])
```

(`prefix_queries` contains "queries", so `NO_DECAY_KEYS` already exempts it from weight decay.)

- [ ] **Step 5: Implement the loss** (`mmae/losses.py`, after `mlm_loss`):

```python
def mlm_emotion_loss(
    logits: torch.Tensor, input_ids: torch.Tensor, token_mask: torch.Tensor,
    emotion_logits: torch.Tensor, emotion: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MLM with the annotator's emotion as one more hidden target per caption (H-b spec 2026-10-07, 5.1): the
    mean cross-entropy over the masked tokens and the emotion slots together. Also returns, for logging only,
    loss_mlm_tokens (mean over masked tokens, 0 if none) and loss_emotion (mean over captions)."""
    if token_mask.any():
        token_ce = F.cross_entropy(logits[token_mask].float(), input_ids[token_mask], reduction="sum")
    else:
        token_ce = logits.float().sum() * 0.0
    emotion_ce = F.cross_entropy(emotion_logits.float(), emotion, reduction="sum")
    n_tokens, n_captions = token_mask.sum(), emotion.shape[0]
    total = (token_ce + emotion_ce) / (n_tokens + n_captions)
    parts = {
        "loss_mlm_tokens": (token_ce / n_tokens.clamp(min=1)).detach(),
        "loss_emotion": (emotion_ce / n_captions).detach(),
    }
    return total, parts
```

- [ ] **Step 6: Wire it into `MultiMAE`** (`mmae/models/model.py`):
  - import: `from mmae.losses import contrastive_loss, mae_loss, mlm_emotion_loss, mlm_loss`
  - module constant below `MLM_IMAGE_SOURCES`: `NUM_EMOTIONS = 9  # ArtELingo's emotion classes (mmae.data.artelingo.EMOTIONS)`
  - in `__init__`, right after `self.text_mode` validation:

```python
        self.emotion_head = bool(cfg.get("emotion_head", False))
        if self.emotion_head and not (self.reconstruction and self.use_text):
            raise ValueError("emotion_head needs the text decoder (reconstruction with the text modality)")
```

  - the text decoder construction becomes:

```python
                self.text_decoder = QueryDecoder(
                    max_text_len, dim, self.text.vocab_size, depth=dec.depth, heads=dec.heads, dropout=dec.dropout,
                    prefix_queries=1 if self.emotion_head else 0, prefix_out_dim=NUM_EMOTIONS,
                )
```

  - in `forward`, replace the MLM block and the output assembly:

```python
        logged: dict[str, torch.Tensor] = {}
        if self.use_text:
            decoded = self.text_decoder(text_memory, text_memory_padding, query_padding=text_padding)
            if self.emotion_head:
                if "emotion" not in batch:
                    raise KeyError("model.emotion_head=true needs batch['emotion'] (ArtELingo data)")
                logits, emotion_logits = decoded
                losses["mlm"], logged = mlm_emotion_loss(
                    logits, input_ids, token_mask, emotion_logits[:, 0], batch["emotion"]
                )
            else:
                losses["mlm"] = mlm_loss(decoded, input_ids, token_mask)

        out = {f"loss_{name}": value for name, value in losses.items()}
        out["loss"] = sum(self.loss_weights[name] * value for name, value in losses.items())
        out.update(logged)  # logged only; not part of the total
        return out
```

  - `configs/model/base.yaml`, after `pooled_conditioning`:

```yaml
emotion_head: false      # H-b (spec 2026-10-07): one extra text-decoder query predicts the annotator's emotion
                         # (9 ArtELingo classes); always a target, never an input. Needs batch['emotion'].
```

- [ ] **Step 7: Run** `python -m pytest tests/test_emotion_head.py tests/test_model_variants.py tests/test_model.py -q`. Expected: all PASS.

- [ ] **Step 8: Confirm two guards bite.** (a) Remove the `raise KeyError` branch so a missing label falls through (e.g. `batch.get("emotion", torch.zeros(...))`): `test_emotion_head_needs_labels` must fail. (b) Feed `batch["emotion"]` into the decoder memory (e.g. add `batch["emotion"].float().view(-1, 1, 1)` to `text_memory`): `test_emotion_is_a_target_never_an_input` must fail. Revert both.

- [ ] **Step 9: Full fast suite** `python -m pytest -q`. Expected: PASS (the slow suite is not run).

- [ ] **Step 10: Change log and commit** `model: emotion head on the caption decoder (H-b spec 2026-10-07, 5.1)`.

---

### Task 3: ArtELingo data module with the AL-28 hold-out

**Files:**
- Create: `mmae/data/artelingo.py`, `mmae/data/al28_paintings.txt`, `scripts/make_al28_heldout.py`, `tests/test_artelingo.py`
- Modify: `tests/helpers.py` (`make_fake_artelingo`), `tests/conftest.py` (`fake_artelingo` fixture)

**Interfaces:**
- Consumes: `mmae.data.coco` nothing (own loader).
- Produces (`mmae/data/artelingo.py`):
  - `EMOTIONS: tuple[str, ...]` (9, order above), `EMOTION_INDEX: dict[str, int]`, `HELDOUT_FILE: Path`.
  - `heldout_paintings(path: str | Path | None = None) -> frozenset[str]` (None = `HELDOUT_FILE`).
  - `load_painting(path: Path) -> PIL.Image.Image` (RGB, JPEG draft decode at no less than 224 px a side).
  - `retrieval_groups(annotations_dir, split, heldout=frozenset(), limit=None) -> list[tuple[str, list[str], list[int]]]`.
  - `ArtelingoPairs(images_dir, annotations_dir, split, transform, limit=None, heldout=frozenset())`, items `(Tensor, str, int)`.
  - `ArtelingoRetrieval(images_dir, annotations_dir, split, transform, limit=None, heldout=frozenset())`, items `(Tensor, list[str])`; attribute `emotions: list[list[int]]`.
  - `tests/helpers.make_fake_artelingo(root: Path) -> tuple[Path, Path, Path]` (images_dir, annotations_dir, heldout_file).

- [ ] **Step 1: Generate the hold-out list.** `scripts/make_al28_heldout.py`:

```python
"""Write mmae/data/al28_paintings.txt: every painting of the public ArtELingo-28 CSV, sorted, one per line.
H-b holds these paintings out of ArtELingo training and validation (spec 2026-10-07, section 4)."""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=DEFAULT_CSV)
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "mmae" / "data" / "al28_paintings.txt"))
args = parser.parse_args()
paintings = sorted(set(pd.read_csv(args.csv, usecols=["painting"])["painting"]))
Path(args.out).write_text("\n".join(paintings) + "\n")
print(f"{len(paintings)} paintings -> {args.out}")
```

Run it: `python scripts/make_al28_heldout.py`. Expected output: `1658 paintings -> .../mmae/data/al28_paintings.txt`.

- [ ] **Step 2: Fake ArtELingo** in `tests/helpers.py`:

```python
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
```

`tests/conftest.py`:

```python
@pytest.fixture
def fake_artelingo(tmp_path):
    return make_fake_artelingo(tmp_path / "artelingo_root")
```

(import `make_fake_artelingo` next to `make_fake_coco`).

- [ ] **Step 3: Write the failing tests** `tests/test_artelingo.py`:

```python
"""ArtELingo data for H-b (spec 2026-10-07, section 4)."""
from pathlib import Path

import pytest
import torch

from helpers import ARTELINGO_EMOTIONS
from mmae.data.artelingo import (
    EMOTIONS, HELDOUT_FILE, ArtelingoPairs, ArtelingoRetrieval, heldout_paintings, load_painting, retrieval_groups,
)
from mmae.data.transforms import build_image_transform
from mmae.models.model import NUM_EMOTIONS

AL28_CSV = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv")


@pytest.fixture(scope="module")
def transform():
    return build_image_transform("openai/clip-vit-base-patch32")


def test_emotion_classes_match_the_model():
    assert EMOTIONS == ARTELINGO_EMOTIONS and len(EMOTIONS) == NUM_EMOTIONS


def test_packaged_heldout_list_is_every_al28_painting():
    paintings = heldout_paintings()
    assert len(paintings) == 1658
    if AL28_CSV.exists():
        import pandas as pd
        assert paintings == frozenset(pd.read_csv(AL28_CSV, usecols=["painting"])["painting"])


def test_train_pairs_drop_heldout_paintings(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    pairs = ArtelingoPairs(images, annotations, "train", transform, heldout=heldout_paintings(heldout_file))
    assert len(pairs) == 10  # 6 paintings x 2 captions, p5 held out
    assert not any(caption.startswith("p5 ") for _, caption, _ in pairs.pairs)
    image, caption, emotion = pairs[0]
    assert image.shape == (3, 224, 224) and isinstance(caption, str) and 0 <= emotion < 9


def test_val_pairs_are_caption_major_and_drop_heldout(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    pairs = ArtelingoPairs(images, annotations, "val", transform, heldout=heldout_paintings(heldout_file))
    paintings = [caption.split()[0] for _, caption, _ in pairs.pairs]
    assert paintings == ["v0", "v1"] * 5  # caption-major, v2 held out
    assert [e for _, c, e in pairs.pairs if c.startswith("v1 ")] == [EMOTIONS.index(ARTELINGO_EMOTIONS[(1 + k) % 9]) for k in range(5)]


def test_retrieval_drops_heldout_from_val_but_not_test(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    heldout = heldout_paintings(heldout_file)
    val = ArtelingoRetrieval(images, annotations, "val", transform, heldout=heldout)
    test = ArtelingoRetrieval(images, annotations, "test", transform, heldout=heldout)
    assert len(val) == 2 and len(test) == 3
    image, captions = test[1]
    assert image.shape == (3, 224, 224) and len(captions) == 5 and captions[0].startswith("t1 ")
    assert test.emotions[1] == [EMOTIONS.index(ARTELINGO_EMOTIONS[(1 + k) % 9]) for k in range(5)]


def test_retrieval_caption_without_emotion_fails(fake_artelingo):
    images, annotations, _ = fake_artelingo
    path = annotations / "artelingo_test.json"
    import json
    items = json.loads(path.read_text())
    path.write_text(json.dumps(items[1:]))  # drop t0's first caption from the per-caption file
    with pytest.raises(KeyError, match="t0"):
        retrieval_groups(annotations, "test")


def test_limit_keeps_the_first_items(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    assert len(ArtelingoPairs(images, annotations, "train", transform, limit=3)) == 3
    assert len(ArtelingoRetrieval(images, annotations, "test", transform, limit=2)) == 2


def test_load_painting_returns_rgb_at_least_224(fake_artelingo):
    images, _, _ = fake_artelingo
    image = load_painting(images / "Style_A" / "p1.jpg")  # grayscale on disk
    assert image.mode == "RGB" and min(image.size) >= 224


@pytest.mark.slow
def test_real_artelingo_splits():
    """The real files: hold-out counts and the retrieval lookups (spec section 4)."""
    annotations = Path("/data/PDD/artelingo")
    heldout = heldout_paintings()
    val = retrieval_groups(annotations, "val", heldout)
    test = retrieval_groups(annotations, "test", heldout)
    assert len(test) == 4975  # lookups complete (retrieval_groups raises otherwise)
    assert len(val) == 2421   # 2,469 val retrieval paintings minus 48 AL-28 paintings
    assert not any(Path(img).stem in heldout for img, _, _ in val)
    train = ArtelingoPairs(annotations.parent / "wikiart_proj" / "wikiart", annotations, "train", lambda x: x, heldout=heldout)
    assert len(train) == 302841  # 308,723 captions minus those of the 1,160 AL-28 train paintings
```

- [ ] **Step 4: Run** `python -m pytest tests/test_artelingo.py -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'mmae.data.artelingo'`.

- [ ] **Step 5: Implement** `mmae/data/artelingo.py`:

```python
"""ArtELingo English (WikiArt paintings) for H-b (spec docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md,
section 4): (image, caption, emotion) pairs and 5-caption retrieval sets. Every ArtELingo-28 painting (the dense
human reference) is held out of train and validation; test keeps them."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # some WikiArt scans exceed PIL's decompression-bomb limit; the data is trusted

log = logging.getLogger(__name__)

EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else")
EMOTION_INDEX = {name: i for i, name in enumerate(EMOTIONS)}
CAPTIONS_PER_IMAGE = 5
SPLIT_FILES = {"train": "artelingo_train.json", "val": "artelingo_val.json", "test": "artelingo_test.json"}
RETRIEVAL_FILES = {"val": "artelingo_val_retrieval.json", "test": "artelingo_test_retrieval.json"}
HELDOUT_FILE = Path(__file__).with_name("al28_paintings.txt")
MIN_SIDE = 224  # CLIP's input size: JPEGs are decoded at a reduced scale no smaller than this


def heldout_paintings(path: str | Path | None = None) -> frozenset[str]:
    """The held-out painting names, one per line (None = the packaged ArtELingo-28 list)."""
    text = Path(path or HELDOUT_FILE).read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def read_json(annotations_dir: str | Path, name: str) -> list[dict]:
    with open(Path(annotations_dir) / name, encoding="utf-8") as f:
        return json.load(f)


def load_painting(path: Path) -> Image.Image:
    """RGB image; JPEGs use PIL's draft mode (DCT scaling to the smallest scale with both sides >= MIN_SIDE), which
    makes large WikiArt scans cheap to decode before CLIP's resize to 224."""
    with Image.open(path) as image:
        image.draft("RGB", (MIN_SIDE, MIN_SIDE))
        return image.convert("RGB")


def emotion_lookup(items: list[dict]) -> dict[tuple[str, str], int]:
    """(painting, caption) -> emotion index from a per-caption file; on a duplicate key the first entry wins and
    the number of such keys is logged."""
    lookup: dict[tuple[str, str], int] = {}
    duplicates = 0
    for item in items:
        key = (item["painting"], item["caption"])
        if key in lookup:
            duplicates += lookup[key] != EMOTION_INDEX[item["emotion"]]
            continue
        lookup[key] = EMOTION_INDEX[item["emotion"]]
    if duplicates:
        log.info("%d (painting, caption) keys carry more than one emotion; the first is kept", duplicates)
    return lookup


def retrieval_groups(
    annotations_dir: str | Path, split: str, heldout: frozenset[str] = frozenset(), limit: int | None = None
) -> list[tuple[str, list[str], list[int]]]:
    """(image path, 5 captions, their 5 emotion indices) per painting of the val or test retrieval file, in file
    order. Held-out paintings are dropped from val only (test keeps them); `limit` keeps the first N paintings."""
    if split not in RETRIEVAL_FILES:
        raise ValueError(f"retrieval sets are {sorted(RETRIEVAL_FILES)}, got {split!r}")
    drop = heldout if split == "val" else frozenset()
    lookup = emotion_lookup(read_json(annotations_dir, SPLIT_FILES[split]))
    groups = []
    for item in read_json(annotations_dir, RETRIEVAL_FILES[split]):
        if item["painting"] in drop:
            continue
        captions = item["caption"][:CAPTIONS_PER_IMAGE]
        if len(captions) < CAPTIONS_PER_IMAGE:
            raise ValueError(f"{item['painting']} has {len(captions)} captions, fewer than {CAPTIONS_PER_IMAGE}")
        emotions = []
        for caption in captions:
            key = (item["painting"], caption)
            if key not in lookup:
                raise KeyError(f"no emotion for {key} in {SPLIT_FILES[split]}")
            emotions.append(lookup[key])
        groups.append((item["image"], list(captions), emotions))
    return groups[:limit]


class ArtelingoPairs(Dataset):
    """One (image, caption, emotion index) item per caption. train: every caption of artelingo_train.json except
    the held-out paintings'; val and test: flattened caption-major from the 5-caption retrieval file (an unshuffled
    batch then shows distinct paintings), held-out paintings dropped from val. `limit` keeps the first N pairs."""

    def __init__(
        self, images_dir: str | Path, annotations_dir: str | Path, split: str,
        transform: Callable[[Image.Image], torch.Tensor], limit: int | None = None,
        heldout: frozenset[str] = frozenset(),
    ) -> None:
        if split == "train":
            items = read_json(annotations_dir, SPLIT_FILES["train"])
            self.pairs = [(it["image"], it["caption"], EMOTION_INDEX[it["emotion"]])
                          for it in items if it["painting"] not in heldout]
        else:
            groups = retrieval_groups(annotations_dir, split, heldout)
            self.pairs = [(image, captions[c], emotions[c])
                          for c in range(CAPTIONS_PER_IMAGE) for image, captions, emotions in groups]
        self.pairs = self.pairs[:limit]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        image, caption, emotion = self.pairs[index]
        return self.transform(load_painting(self.images_dir / image)), caption, emotion


class ArtelingoRetrieval(Dataset):
    """One item per painting with its 5 captions (val or test retrieval file); `emotions` holds their labels."""

    def __init__(
        self, images_dir: str | Path, annotations_dir: str | Path, split: str,
        transform: Callable[[Image.Image], torch.Tensor], limit: int | None = None,
        heldout: frozenset[str] = frozenset(),
    ) -> None:
        groups = retrieval_groups(annotations_dir, split, heldout, limit)
        self.items = [(image, captions) for image, captions, _ in groups]
        self.emotions = [emotions for _, _, emotions in groups]
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        image, captions = self.items[index]
        return self.transform(load_painting(self.images_dir / image)), list(captions)
```

- [ ] **Step 6: Run** `python -m pytest tests/test_artelingo.py -q`, then the real-data check `python -m pytest tests/test_artelingo.py -q -m slow -k real` (CPU, reads the JSONs only). Expected: all PASS.

- [ ] **Step 7: Confirm the hold-out guard bites.** Make `ArtelingoPairs` ignore `heldout` for train: `test_train_pairs_drop_heldout_paintings` fails. Revert.

- [ ] **Step 8: Commit** `data: ArtELingo pairs and retrieval sets with the AL-28 hold-out (H-b spec 2026-10-07, 4)` (include `mmae/data/al28_paintings.txt` and the script).

---

### Task 4: Wiring: collate, dataset factory, configs, extended-metrics guard, CPU smoke of every arm

**Files:**
- Create: `mmae/data/factory.py`, `configs/data/artelingo.yaml`, `configs/data/artelingo_cluster.yaml`, `tests/test_artelingo_wiring.py`
- Modify: `mmae/data/collate.py` (`Collator.pairs`), `mmae/data/__init__.py`, `mmae/engine/trainer.py` (dataset construction), `evaluate.py` (dataset construction), `mmae/engine/eccv.py` (`build_extended_metrics`), `tests/helpers.py` (`run_train_artelingo`)

**Interfaces:**
- Consumes: Task 2 (`batch["emotion"]`), Task 3 (`ArtelingoPairs`, `ArtelingoRetrieval`, `heldout_paintings`).
- Produces:
  - `mmae.data.factory.dataset_name(dcfg) -> str` ("coco" when `name` is absent), `build_pairs(dcfg, split, transform, limit) -> Dataset`, `build_retrieval(dcfg, split, transform, limit) -> Dataset`.
  - Batches from `Collator.pairs` carry `emotion` (B,) long when items are 3-tuples.
  - `data=artelingo` / `data=artelingo_cluster` configs with keys `name, images_dir, annotations_dir, heldout_file, max_text_len, limit_train, limit_val, limit_test`, and `eval.extended_metrics: false`.
  - `tests/helpers.run_train_artelingo(cwd, fake_artelingo, *overrides, script="train.py")`.

- [ ] **Step 1: Write the failing tests** `tests/test_artelingo_wiring.py`:

```python
"""ArtELingo wiring: collate, dataset factory, configs, and a CPU smoke of every D4 arm (spec 2026-10-07)."""
import json

import pytest
import torch

from helpers import compose_cfg, run_train_artelingo
from mmae.data import Collator
from mmae.data.factory import build_pairs, build_retrieval, dataset_name
from mmae.data.transforms import build_image_transform

ARMS = {
    "C": ("model=contrastive",),
    "ML-80": ("model=fusion_multilearner", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0", "model.emotion_head=true"),
    "ML-80+MAE": ("model=fusion_multilearner", "model.masking.text_ratio=0.8", "model.emotion_head=true"),
    "Par-cap": ("model=fusion_multilearner", "model.masking.text_ratio=1.0", "model.mlm_image_source=clean",
                "model.loss.weights.mae=0", "model.emotion_head=true"),
}


def test_configs():
    local, cluster = compose_cfg("data=artelingo"), compose_cfg("data=artelingo_cluster")
    assert set(local.data) == set(cluster.data)
    assert local.data.name == "artelingo" and local.data.max_text_len == 40 == cluster.data.max_text_len
    assert local.eval.extended_metrics is False and cluster.eval.extended_metrics is False
    assert cluster.data.images_dir == "/local/wding/Dataset/wikiart_proj/wikiart"
    assert cluster.data.annotations_dir == "/local/wding/Dataset/artelingo"
    assert cluster.paths.res_dir == "/local/wding/res/MultiMAE/artelingo"
    assert dataset_name(compose_cfg().data) == "coco"  # COCO configs carry no name


def test_collate_adds_emotion_for_artelingo_pairs():
    collator = Collator("openai/clip-vit-base-patch32", 40)
    image = torch.zeros(3, 224, 224)
    batch = collator.pairs([(image, "a calm sea", 2), (image, "an angry sky", 4)])
    assert batch["emotion"].tolist() == [2, 4] and batch["emotion"].dtype == torch.long
    assert "emotion" not in collator.pairs([(image, "a dog"), (image, "a cat")])


def test_factory_builds_both_datasets(fake_artelingo, fake_coco):
    images, annotations, heldout = fake_artelingo
    transform = build_image_transform("openai/clip-vit-base-patch32")
    dcfg = compose_cfg("data=artelingo", f"data.images_dir={images}", f"data.annotations_dir={annotations}",
                       f"data.heldout_file={heldout}").data
    assert len(build_pairs(dcfg, "train", transform, None)) == 10
    assert len(build_retrieval(dcfg, "val", transform, None)) == 2
    coco_images, coco_annotations = fake_coco
    ccfg = compose_cfg(f"data.images_dir={coco_images}", f"data.annotations_dir={coco_annotations}").data
    assert len(build_pairs(ccfg, "train", transform, None)) == 16


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_every_arm_trains_on_artelingo(arm, tmp_path, fake_artelingo):
    result = run_train_artelingo(tmp_path, fake_artelingo, *ARMS[arm], "train.seeded_sampler=true")
    assert result.returncode == 0, result.stderr[-3000:]
    (run_json,) = list((tmp_path / "res").rglob("run.json"))
    run = json.loads(run_json.read_text())
    assert run["status"] == "completed"
    test = run["results"]["test"]
    assert "test/retrieval/rsum" in test
    assert not any(k.startswith("test/eccv") for k in test)
    if arm != "C":
        assert "test/loss_emotion" in test and "test/loss_mlm_tokens" in test
```

In `tests/helpers.py` add (next to `run_train`):

```python
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
```

(Check `run.json`'s real layout first with an existing COCO run under `res/coco/multimae/ml_improve/*/run.json` and adapt the `results`/`test` keys in the test if they differ; `stage2_table.py` reads `rj["results"]["test"]["test/eccv/map_at_r"]`.)

- [ ] **Step 2: Run** `python -m pytest tests/test_artelingo_wiring.py -q -x`. Expected: FAIL (`No module named 'mmae.data.factory'`).

- [ ] **Step 3: Collate.** In `mmae/data/collate.py`:

```python
    def pairs(self, batch: list[tuple]) -> dict[str, torch.Tensor]:
        """Items are (image, caption) or, for ArtELingo, (image, caption, emotion index)."""
        images, captions, *rest = zip(*batch)
        out = {"pixel_values": torch.stack(images), **self.tokenize(list(captions))}
        if rest:
            out["emotion"] = torch.tensor(rest[0], dtype=torch.long)
        return out
```

- [ ] **Step 4: Factory** `mmae/data/factory.py`:

```python
"""Dataset selection by data.name: coco (the default when the key is absent, as in every COCO config) or
artelingo (H-b spec 2026-10-07)."""
from __future__ import annotations

from typing import Callable

from omegaconf import DictConfig
from torch.utils.data import Dataset

from mmae.data.artelingo import ArtelingoPairs, ArtelingoRetrieval, heldout_paintings
from mmae.data.coco import CocoPairs, CocoRetrieval

DATASETS = ("coco", "artelingo")


def dataset_name(dcfg: DictConfig) -> str:
    name = str(dcfg.get("name", "coco"))
    if name not in DATASETS:
        raise ValueError(f"data.name must be one of {DATASETS}, got {name!r}")
    return name


def build_pairs(dcfg: DictConfig, split: str, transform: Callable, limit: int | None) -> Dataset:
    if dataset_name(dcfg) == "artelingo":
        return ArtelingoPairs(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit,
                              heldout=heldout_paintings(dcfg.get("heldout_file")))
    return CocoPairs(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit)


def build_retrieval(dcfg: DictConfig, split: str, transform: Callable, limit: int | None) -> Dataset:
    if dataset_name(dcfg) == "artelingo":
        return ArtelingoRetrieval(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit,
                                  heldout=heldout_paintings(dcfg.get("heldout_file")))
    return CocoRetrieval(dcfg.images_dir, dcfg.annotations_dir, split, transform, limit)
```

Export `build_pairs`, `build_retrieval`, `dataset_name` from `mmae/data/__init__.py` (update its docstring to "COCO and ArtELingo datasets, ...").

- [ ] **Step 5: Use the factory.** In `mmae/engine/trainer.py`: the train set becomes `build_pairs(dcfg, "train", self.transform, dcfg.limit_train)`; in `_eval_loaders`, `CocoPairs(...)` becomes `build_pairs(d, split, self.transform, limit)` and `CocoRetrieval(...)` becomes `build_retrieval(d, split, self.transform, limit)`. In `evaluate.py`, `CocoRetrieval(...)` becomes `build_retrieval(data, split, build_image_transform(processor), data.limit_val if split == "val" else data.limit_test)`. Update the imports.

- [ ] **Step 6: Guard the extended metrics.** In `mmae/engine/eccv.py::build_extended_metrics`, right after the first `return None` check:

```python
    if dataset_name(cfg.data) != "coco":
        return None  # ECCV Caption, CxC, COCO 1K and PMRP exist for COCO only
```

(import `dataset_name` from `mmae.data.factory`; if that import is circular, import inside the function.)

- [ ] **Step 7: Configs.** `configs/data/artelingo.yaml`:

```yaml
# @package _global_
# ArtELingo English (WikiArt paintings) for H-b (spec docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md).
# Selected with data=artelingo; this file replaces data/coco.yaml. The ArtELingo-28 paintings are held out of train
# and validation (data.heldout_file; null = mmae/data/al28_paintings.txt). Extended COCO metrics do not apply.
data:
  name: artelingo
  images_dir: /data/PDD/wikiart_proj/wikiart
  annotations_dir: /data/PDD/artelingo   # artelingo_{train,val,test}.json, artelingo_{val,test}_retrieval.json
  heldout_file: null
  max_text_len: 40     # with BOS and EOS, 2.6% of train captions exceed 32 tokens and 0.02% exceed 40
  limit_train: null
  limit_val: null
  limit_test: null
eval:
  extended_metrics: false
```

`configs/data/artelingo_cluster.yaml`: the same keys with

```yaml
  images_dir: /local/wding/Dataset/wikiart_proj/wikiart
  annotations_dir: /local/wding/Dataset/artelingo
```

plus

```yaml
paths:
  res_dir: /local/wding/res/MultiMAE/artelingo   # strictly under the cluster profile's RESULTS_REMOTE; pulls to res/artelingo/
```

and a header saying it repeats every key of `artelingo.yaml` (tested) and that the cluster-run skill keeps a shared copy.

- [ ] **Step 8: Run** `python -m pytest tests/test_artelingo_wiring.py -q` and then the full fast suite `python -m pytest -q`. Expected: all PASS. If a COCO-only assumption elsewhere breaks the ArtELingo smoke (e.g. code reading `cfg.data.pm_dir`), read it through `cfg.data.get(...)` or guard it with `dataset_name`, and add the file to the change log.

- [ ] **Step 9: Confirm the extended-metrics guard bites.** Remove the `dataset_name` check and run `python -m pytest tests/test_artelingo_wiring.py -q -k "trains_on_artelingo and C"` with `eval.extended_metrics=true` appended to that arm's overrides in a scratch copy of the test: it must fail. Revert.

- [ ] **Step 10: Change log and commit** `cluster run: ArtELingo data path for H-b D4 (factory, configs, collate emotion)`.

---

### Task 5 (controller): real-data smoke, timing, cluster checks, wave-1 launch

Not delegated; the main session runs it.

- [ ] **Step 1: Local GPU smoke.** Check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` (empty) and run under the lock, from the repo root:

```bash
flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/MultiMAE/bin/python train.py data=artelingo \
  model=fusion_multilearner model.masking.text_ratio=0.8 model.loss.weights.mae=0 model.emotion_head=true \
  data.limit_train=4096 data.limit_val=200 data.limit_test=200 train.epochs=1 train.log_every=8 \
  wandb.mode=disabled paths.res_dir=<scratchpad>/hb_smoke
```

Pass: exit 0; `train/loss_emotion` in `metrics.jsonl` falls below ln 9 = 2.197 by the end of the epoch; `val/retrieval/rsum` present. Record items per second for the timing estimate. Repeat for Par-cap (`model.masking.text_ratio=1.0 model.mlm_image_source=clean`) and C (`model=contrastive`) with `data.limit_train=1024`.

- [ ] **Step 2: Cluster config.** `cluster configs --adopt configs/data/artelingo_cluster.yaml`. If `IMAGE_ANNOTATION_GLOB` in `~/.claude/skills/cluster-run/projects/MultiMAE.conf` must cover ArtELingo for `ready.data`, extend it to `coco_karpathy_*.json artelingo_train.json artelingo_val*.json artelingo_test*.json` (a local profile edit; note it in the registry).

- [ ] **Step 3: Sync and check.** `cluster sync`, then for each arm `cluster check --node node404 --sync-data -- python train.py data=artelingo_cluster <arm overrides> train.seeded_sampler=true seed=42 wandb.group=hb_d4 wandb.name=<arm>`; all four verdicts yes. Repeat the data check on node405 and node411 (data syncs per node).

- [ ] **Step 4: Launch wave 1** (9 jobs, one GPU each): seeds 42 and 43 of C, ML-80, ML-80+MAE, Par-cap, and seed 44 of ML-80, spread over node404, node405, node411 (`cluster launch --node <node> -- python train.py ...`). Record each tag in `tests/20261007_hb/runs.md`. Watch with `cluster watch <tag>` in the background (one watcher per job, or a queue script adapted from `tests/20261003_ml_improve/queue.sh`); pull each run when it succeeds.

- [ ] **Step 5: First-epoch timing.** From W&B or `cluster watch <tag> --once`, compare the first epoch's wall time with the 4 h (masked) and 2.3 h (contrastive) estimates; update the registry.

- [ ] **Step 6: Gate before wave 2** (spec section 13): needs Plan 2's readout; seed 44 of C, ML-80+MAE and Par-cap launch only after it passes.
