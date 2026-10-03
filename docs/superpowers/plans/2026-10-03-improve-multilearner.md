# Improving fusion_multilearner: Stage 0 to 2 code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the default-off switches, evaluation and diagnostics that the staged multilearner experiments need: M1 (MLM reads the clean image), M2b (content-word masking), M3 (cross-modal pooled conditioning), M6 (masked-view InfoNCE), R2 (per-tower learning rates, layer decay, frozen vision warmup), a seeded training sampler, VWSD evaluation and the Stage 0 diagnostics.

**Architecture:** Every change is a config switch whose default reproduces today's behaviour bit for bit, so the 12 baseline runs stay valid references and the existing fast tests pass unchanged. Model switches live in `mmae/models/model.py` (with small helpers in `backbones.py`, `masking.py`, `fusion.py`), training switches in `mmae/engine/trainer.py`, evaluation in a new `mmae/engine/vwsd.py` and `mmae/engine/diagnostics.py` with a cluster entry `scripts/diagnose.py` behind `scripts/run_diagnostics.sh`.

**Tech Stack:** Python 3.11, PyTorch 2.11, HF transformers CLIP, Hydra/OmegaConf, Accelerate, pytest (CPU, tiny random CLIP, fake COCO).

**Spec:** `docs/superpowers/specs/2026-10-03-improve-multilearner-design.md` (sections 5 to 8). Read it with this plan.

## Global Constraints

- Python and tests: `/root/miniconda3/envs/MultiMAE/bin/python`; run tests on CPU only: `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/MultiMAE/bin/python -m pytest -q <tests>`. Never use the local GPU (DAS6 only).
- Defaults keep today's behaviour exactly: with every new key at its default, losses, masks, parameter groups and data order are unchanged, and every existing test passes without edits.
- Old run configs (written before these keys existed) must still build: read every new **model** key with `cfg.get(key, default)` / `cfg.masking.get(...)` / `cfg.loss.weights.get(...)`, and every new **train** key with `cfg.train.get(...)`.
- Every new train key goes into both `configs/train/default.yaml` and `configs/train/debug.yaml` (debug replaces default).
- Do not edit `configs/data/coco_cluster.yaml` (a shared copy lives in the cluster skill). New data-like paths go under `eval.` in `configs/config.yaml`.
- Every trainable parameter must get a gradient in every config (DDP runs with `find_unused_parameters=False`).
- For each new guard test, break the guarded code on purpose once, confirm the test fails, restore, and say so in your report.
- Change log: for every edit to an existing source file, append to `.claude/20261003_log.md` a `# <path>` header, a short before/after snippet and why (one file per day, appended; create it if missing).
- Commits: `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit`, message ending with the two lines
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01QDdDeHpDdoi9UTFWhAEeXK`. Do not push.
- At the end of every task the full fast suite passes: `python -m pytest -q` (CPU, several minutes).

## Review Focus

1. Evaluating one of the 12 old baseline runs (`evaluate.py eval.run_dir=...`): its `config.yaml` has none of the new keys; the model must build and load its checkpoint with today's behaviour. Pinned in Task 2 (`test_model_builds_from_a_config_without_the_new_keys`).
2. A caption with no content word (only stop words, or empty) under `text_mode=content`: no token is masked in it, the batch's losses stay finite. Pinned in Task 3.
3. `freeze_vision_epochs` with `grad_accum > 1`: the frozen window counts optimizer steps, not batches, so the vision tower is still exactly unchanged after the frozen epoch. Pinned in Task 6.
4. Every new model switch combined with `pooling=mean`: shapes work and every trainable parameter gets a gradient. Pinned by the `VARIANTS x pooling` gradient test in Tasks 2 to 5.
5. VWSD images that are PNG with alpha, palette or grayscale: they are converted to RGB before the CLIP transform. Pinned in Task 8 (the fake VWSD tree includes RGBA and L images).

---

### Task 1: Tower `pool()` methods (refactor, no behaviour change)

**Files:**
- Modify: `mmae/models/backbones.py` (`ClipVisionTower.embed`, `ClipTextTower.embed`)
- Test: `tests/test_backbones.py` (append)

**Interfaces:**
- Produces: `ClipVisionTower.pool(tokens: Tensor) -> Tensor` (B, E) L2-normalized, from `encode()`'s hidden states of a clean image; `ClipTextTower.pool(tokens: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor` (B, E) L2-normalized. `embed(x) == pool(encode(x))` exactly.

- [ ] **Step 1: Write the failing test** (append to `tests/test_backbones.py`)

```python
@pytest.mark.parametrize("pooling", ["native", "mean"])
def test_pool_of_encode_is_embed(pooling, tokenizer):
    from helpers import make_batch
    from mmae.models.backbones import build_backbone

    torch.manual_seed(0)
    towers = build_backbone("hf_clip", "tiny-random-clip", pooling)
    batch = make_batch(tokenizer)
    with torch.no_grad():
        image = towers.vision.pool(towers.vision.encode(batch["pixel_values"]))
        text_tokens = towers.text.encode(batch["input_ids"], batch["attention_mask"])
        text = towers.text.pool(text_tokens, batch["input_ids"], batch["attention_mask"])
        assert torch.equal(image, towers.vision.embed(batch["pixel_values"]))
        assert torch.equal(text, towers.text.embed(batch["input_ids"], batch["attention_mask"]))
```

If `tests/test_backbones.py` does not import `pytest` and `torch` at the top already, add the imports.

- [ ] **Step 2: Run it, expect FAIL** — `python -m pytest -q tests/test_backbones.py -k pool_of_encode` → `AttributeError: 'ClipVisionTower' object has no attribute 'pool'`.

- [ ] **Step 3: Implement.** In `ClipVisionTower` replace `embed` with:

```python
    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2-normalized joint-space embedding from encode()'s hidden states of a clean image."""
        if self.pooling == "native":
            z = self.projection(self.model.post_layernorm(tokens[:, 0]))
        else:
            z = self.mean_projection(tokens.mean(dim=1))
        return F.normalize(z, dim=-1)

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encode(pixel_values))
```

In `ClipTextTower` replace `embed` with:

```python
    def pool(self, tokens: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """L2-normalized joint-space embedding from encode()'s hidden states (EOS token, or the mean over real
        tokens under mean pooling). Works on a masked caption's hidden states too: EOS is never masked."""
        if self.pooling == "native":
            rows = torch.arange(tokens.shape[0], device=tokens.device)
            z = self.projection(tokens[rows, self.eos_positions(input_ids).to(tokens.device)])
        else:
            weights = attention_mask.unsqueeze(-1).to(tokens.dtype)
            z = self.mean_projection((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0))
        return F.normalize(z, dim=-1)

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encode(input_ids, attention_mask), input_ids, attention_mask)
```

- [ ] **Step 4: Run** `python -m pytest -q tests/test_backbones.py tests/test_model.py` → PASS (the backbone tests still pin the towers to HF's outputs).

- [ ] **Step 5: Change log + commit** — append the `.claude/20261003_log.md` entry, then `git add mmae/models/backbones.py tests/test_backbones.py` and commit `refactor(models): split tower embed into encode + pool`.

---

### Task 2: M1, the MLM decoder reads the clean image (`model.mlm_image_source`)

**Files:**
- Modify: `configs/model/base.yaml`, `mmae/models/model.py` (`__init__`, `forward`)
- Create: `tests/test_model_variants.py`

**Interfaces:**
- Consumes: Task 1's `vision.pool`, `text.pool`.
- Produces: `MLM_IMAGE_SOURCES = ("masked", "clean", "clean_detached")` in `mmae/models/model.py`; attribute `MultiMAE.mlm_image_source: str`. In `forward`, local names later tasks use: `image_emb`, `text_emb` (clean L2-normalized embeddings, `None` without the contrastive pass), `clean_image_tokens`, `masked_text_hidden` (text tower hidden states of the masked caption, before `text_proj`), `fused` (fusion of masked image + masked text, read by the image decoder) and `text_fused` (what the text decoder reads; `fused` unless M1 is on). `tests/test_model_variants.py` defines `VARIANTS: dict[str, tuple[str, ...]]`, `change(batch, model, masks, modality, masked_part)` and the variant tests that Tasks 3 to 5 extend.

- [ ] **Step 1: Add the config key** to `configs/model/base.yaml`, after `pooling:`:

```yaml
mlm_image_source: masked # masked | clean | clean_detached: the image the MLM decoder reads. masked = the 25% of
                         # patches the masked pass keeps (default); clean = all patches of the clean pass;
                         # clean_detached = the same without gradient into the vision tower through that path
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_model_variants.py`:

```python
"""Default-off model switches of the multilearner line (spec 2026-10-03, section 8): M1, M2b, M3, M6."""
import pytest
import torch
from omegaconf import OmegaConf

from helpers import compose_cfg, make_batch
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP
from test_model import recorder, tiny_model

# name -> overrides; Tasks 3 to 5 add their switches here.
VARIANTS = {
    "m1_clean": ("model.mlm_image_source=clean",),
    "m1_clean_detached": ("model.mlm_image_source=clean_detached",),
}


def variant_batch(tokenizer, variant: str) -> dict:
    return make_batch(tokenizer)


def change(batch: dict, model: MultiMAE, masks: dict, modality: str, masked_part: bool) -> dict:
    """Replace one modality's masked (or visible) content with noise; the other modality is left alone."""
    new = dict(batch)
    if modality == "image":
        ps = model.vision.patch_size
        grid = 224 // ps
        region = masks["patch"].view(-1, 1, grid, grid).float()
        region = region.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3).bool()
        region = region if masked_part else ~region
        new["pixel_values"] = torch.where(region, torch.randn_like(batch["pixel_values"]) * 50, batch["pixel_values"])
    else:
        token_mask = masks["token"]
        visible = batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool() & ~token_mask
        region = token_mask if masked_part else visible
        new["input_ids"] = torch.where(region, torch.randint(1000, 40000, batch["input_ids"].shape), batch["input_ids"])
    return new


@pytest.mark.parametrize("pooling", ["native", "mean"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_gives_every_trainable_parameter_a_gradient(variant, pooling, tokenizer):
    model = tiny_model("fusion_multilearner", f"model.pooling={pooling}", *VARIANTS[variant])
    out = model(variant_batch(tokenizer, variant))
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_decoder_ignores_its_own_masked_content(variant, tokenizer, monkeypatch):
    """No switch may leak a modality's masked content into the decoder that reconstructs it."""
    model = tiny_model("fusion_multilearner", *VARIANTS[variant]).eval()
    masks, run = recorder(model, monkeypatch)
    batch = variant_batch(tokenizer, variant)
    base = run(batch)[0]
    assert torch.equal(run(change(batch, model, masks, "image", True))[0]["image_decoder"], base["image_decoder"])
    assert torch.equal(run(change(batch, model, masks, "text", True))[0]["text_decoder"], base["text_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "image", False))[0]["image_decoder"], base["image_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "text", False))[0]["text_decoder"], base["text_decoder"])


def test_model_builds_from_a_config_without_the_new_keys():
    """Run configs saved before these switches existed (the 12 baseline runs) still build, with the defaults."""
    cfg = compose_cfg("model=fusion_multilearner", f"model.backbone.pretrained={TINY_CLIP}")
    old = OmegaConf.to_container(cfg.model)
    for key in ("mlm_image_source", "pooled_conditioning"):
        old.pop(key, None)
    old["masking"].pop("text_mode", None)
    old["loss"]["weights"].pop("masked_view", None)
    model = MultiMAE(OmegaConf.create(old), max_text_len=cfg.data.max_text_len)
    assert model.mlm_image_source == "masked"


@pytest.mark.parametrize("source", ["clean", "clean_detached"])
def test_m1_text_decoder_reads_the_masked_patches(source, tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", f"model.mlm_image_source={source}").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[0]
    moved = run(change(batch, model, masks, "image", True))[0]
    assert not torch.equal(moved["text_decoder"], base["text_decoder"])


@pytest.mark.parametrize(("source", "reaches_vision"), [("clean", True), ("clean_detached", False)])
def test_m1_detached_stops_the_mlm_gradient_into_the_vision_tower(source, reaches_vision, tokenizer):
    model = tiny_model("fusion_multilearner", f"model.mlm_image_source={source}")
    model(make_batch(tokenizer))["loss_mlm"].backward()
    grad = model.vision.model.encoder.layers[0].mlp.fc1.weight.grad
    assert (grad is not None and grad.abs().sum() > 0) == reaches_vision


@pytest.mark.parametrize("name", ["fusion_none", "contrastive"])
def test_m1_needs_a_fusion_that_reaches_the_text_decoder(name):
    with pytest.raises(ValueError, match="mlm_image_source"):
        tiny_model(name, "model.mlm_image_source=clean")


def test_m1_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="mlm_image_source"):
        tiny_model("fusion_multilearner", "model.mlm_image_source=sideways")
```

- [ ] **Step 3: Run, expect FAIL** — `python -m pytest -q tests/test_model_variants.py` → the M1 tests fail (`mlm_image_source` has no effect, no ValueError).

- [ ] **Step 4: Implement** in `mmae/models/model.py`:

Add next to `MODALITIES`:

```python
MLM_IMAGE_SOURCES = ("masked", "clean", "clean_detached")
```

In `__init__`, after `self.norm_pix = ...`:

```python
        # .get: run configs saved before a switch existed mean its default (today's behaviour)
        self.mlm_image_source = str(cfg.get("mlm_image_source", "masked"))
        if self.mlm_image_source not in MLM_IMAGE_SOURCES:
            raise ValueError(f"mlm_image_source must be one of {MLM_IMAGE_SOURCES}, got {self.mlm_image_source!r}")
        if self.mlm_image_source != "masked" and not (
            self.reconstruction and self.has_contrastive and cfg.fusion.type != "none"
        ):
            raise ValueError(
                "mlm_image_source=clean* needs both modalities, reconstruction and a fusion through which the "
                "text decoder reads the image (concat or multilearner)"
            )
```

Replace the body of `forward` from `if self.has_contrastive:` to the end with:

```python
        image_emb = text_emb = clean_image_tokens = None
        if self.has_contrastive:
            clean_image_tokens = self.vision.encode(images)
            image_emb = self.vision.pool(clean_image_tokens)
            text_emb = self.embed_text(input_ids, attention_mask)
            losses["contrastive"] = contrastive_loss(image_emb, text_emb, self.logit_scale, self.gather)

        if not self.reconstruction:
            return {"loss": self.loss_weights["contrastive"] * losses["contrastive"],
                    "loss_contrastive": losses["contrastive"]}

        image_tokens = text_tokens = text_padding = masked_text_hidden = None
        if self.use_image:
            ids_keep, patch_mask = random_patch_mask(
                images.shape[0], self.vision.num_patches, self.image_ratio, device=images.device
            )
            image_tokens = self.image_proj(self.vision.encode(images, ids_keep))
        if self.use_text:
            token_mask = random_token_mask(attention_mask, batch["special_tokens_mask"], self.text_ratio)
            masked_text_hidden = self.text.encode(input_ids, attention_mask, token_mask)
            text_tokens = self.text_proj(masked_text_hidden)
            text_padding = ~attention_mask.bool()

        fused = self.fusion(image_tokens, text_tokens, text_padding)
        text_fused = fused
        if self.mlm_image_source != "masked":  # M1: the MLM decoder reads every patch of the clean pass
            clean = clean_image_tokens.detach() if self.mlm_image_source == "clean_detached" else clean_image_tokens
            text_fused = self.fusion(self.image_proj(clean), text_tokens, text_padding)
        if self.use_image:
            pred = self.image_decoder(fused.image_memory, fused.image_padding)
            losses["mae"] = mae_loss(pred, images, patch_mask, self.vision.patch_size, norm_pix=self.norm_pix)
        if self.use_text:
            logits = self.text_decoder(text_fused.text_memory, text_fused.text_padding, query_padding=text_padding)
            losses["mlm"] = mlm_loss(logits, input_ids, token_mask)

        out = {f"loss_{name}": value for name, value in losses.items()}
        out["loss"] = sum(self.loss_weights[name] * value for name, value in losses.items())
        return out
```

(`self.vision.encode` then `self.vision.pool` is exactly what `embed_image` did, so the default path is unchanged.)

- [ ] **Step 5: Run** `python -m pytest -q tests/test_model_variants.py tests/test_model.py tests/test_contrastive.py` → PASS. Break the guard once (e.g. drop `.detach()` from the `clean_detached` branch) and confirm `test_m1_detached_stops_the_mlm_gradient_into_the_vision_tower` fails; restore.

- [ ] **Step 6: Full suite, change log, commit** — `python -m pytest -q` → PASS; log entries for `model.py` and `base.yaml`; commit `feat(model): M1 switch, MLM decoder reads the clean image`.

---

### Task 3: M2b, content-word masking (`model.masking.text_mode`)

**Files:**
- Create: `mmae/data/stopwords.py`
- Modify: `mmae/models/masking.py` (`random_token_mask`), `mmae/data/collate.py` (`Collator`), `mmae/engine/trainer.py` (Collator construction), `mmae/models/model.py`, `configs/model/base.yaml`, `tests/helpers.py` (`add_content_mask`)
- Test: `tests/test_masking.py`, `tests/test_data.py`, `tests/test_model_variants.py` (append)

**Interfaces:**
- Produces: `STOP_WORDS: frozenset[str]` and `content_token_table(tokenizer) -> torch.BoolTensor` (vocab,) in `mmae/data/stopwords.py`; `is_content_word(word: str) -> bool`; `random_token_mask(attention_mask, special_tokens_mask, ratio, generator=None, allowed=None)`; `Collator(tokenizer_name, max_text_len, content_words=False)` whose batches gain `content_tokens_mask` (B, T) bool when `content_words`; `MultiMAE.text_mode: str`; `helpers.add_content_mask(batch, tokenizer) -> dict`.

- [ ] **Step 1: Config key.** In `configs/model/base.yaml` under `masking:` add:

```yaml
  text_mode: random      # random | content: content draws the masked tokens only from content words (stop words,
                         # punctuation and numbers never), the same count as random (M2b)
```

- [ ] **Step 2: Failing tests.** Append to `tests/test_masking.py`:

```python
def test_token_mask_allowed_none_is_unchanged():
    attention = torch.ones(4, 12, dtype=torch.long)
    special = torch.zeros(4, 12, dtype=torch.long)
    special[:, 0] = special[:, -1] = 1
    a = random_token_mask(attention, special, 0.4, generator=torch.Generator().manual_seed(3))
    b = random_token_mask(attention, special, 0.4, generator=torch.Generator().manual_seed(3), allowed=None)
    assert torch.equal(a, b)


def test_token_mask_draws_only_allowed_tokens_with_the_full_count():
    attention = torch.ones(3, 12, dtype=torch.long)
    special = torch.zeros(3, 12, dtype=torch.long)
    special[:, 0] = special[:, -1] = 1
    allowed = torch.zeros(3, 12, dtype=torch.bool)
    allowed[0, 1:9] = True      # 8 allowed of 10 real tokens: k = round(0.4 * 10) = 4
    allowed[1, 1:3] = True      # 2 allowed: k capped at 2
    # row 2: nothing allowed -> no mask
    mask = random_token_mask(attention, special, 0.4, generator=torch.Generator().manual_seed(0), allowed=allowed)
    assert not (mask & ~allowed).any()
    assert mask.sum(dim=1).tolist() == [4, 2, 0]
```

(Use the import style already at the top of `tests/test_masking.py`.) Append to `tests/test_data.py`:

```python
def test_collator_marks_content_tokens(tmp_path):
    from mmae.data import Collator
    from helpers import CLIP_NAME

    collator = Collator(CLIP_NAME, 16, content_words=True)
    batch = collator.tokenize(["a dog is running on the beach .", "the of and"])
    content = batch["content_tokens_mask"]
    words = [collator.tokenizer.convert_ids_to_tokens(row) for row in batch["input_ids"].tolist()]
    marked = [[w for w, c in zip(ws, cs) if c] for ws, cs in zip(words, content.tolist())]
    assert marked[0] == ["dog</w>", "running</w>", "beach</w>"]
    assert marked[1] == []
    assert not (content & ~batch["attention_mask"].bool()).any()
    assert not (content & batch["special_tokens_mask"].bool()).any()
    assert "content_tokens_mask" not in Collator(CLIP_NAME, 16).tokenize(["a dog"])
```

Add to `tests/helpers.py`:

```python
def add_content_mask(batch: dict, tokenizer) -> dict:
    """The content_tokens_mask a Collator(content_words=True) would add to a make_batch() batch."""
    from mmae.data.stopwords import content_token_table

    table = content_token_table(tokenizer)
    content = table[batch["input_ids"]] & batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    return {**batch, "content_tokens_mask": content}
```

In `tests/test_model_variants.py` add `"m2b_content": ("model.masking.text_mode=content",),` to `VARIANTS`, make `variant_batch` return `add_content_mask(make_batch(tokenizer), tokenizer)` when `variant == "m2b_content"` (import `add_content_mask` from helpers), and append:

```python
def test_m2b_masks_only_content_words(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content").eval()
    masks, run = recorder(model, monkeypatch)
    batch = add_content_mask(make_batch(tokenizer), tokenizer)
    run(batch)
    assert masks["token"].any()
    assert not (masks["token"] & ~batch["content_tokens_mask"]).any()


def test_m2b_caption_without_content_words_is_finite(tokenizer):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content")
    batch = add_content_mask(make_batch(tokenizer, captions=["the of and", "", "a dog", "on the"]), tokenizer)
    out = model(batch)
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()


def test_m2b_without_the_content_mask_fails_clearly(tokenizer):
    model = tiny_model("fusion_multilearner", "model.masking.text_mode=content")
    with pytest.raises(KeyError, match="content_tokens_mask"):
        model(make_batch(tokenizer))
```

Extend `test_model_builds_from_a_config_without_the_new_keys` with `assert model.text_mode == "random"`.

- [ ] **Step 3: Run, expect FAIL** — `python -m pytest -q tests/test_masking.py tests/test_data.py tests/test_model_variants.py`.

- [ ] **Step 4: Implement.** Create `mmae/data/stopwords.py`:

```python
"""Content words for content-word masking (M2b): a token is a content token when its word piece is alphabetic
and not an English stop word. Stop words are NLTK's English list (179 words)."""
from __future__ import annotations

import torch

STOP_WORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him his
himself she she's her hers herself it it's its itself they them their theirs themselves what which who whom
this that that'll these those am is are was were be been being have has had having do does did doing a an the
and but if or because as until while of at by for with about against between into through during before after
above below to from up down in out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too very s t can will just don
don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn
hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't shouldn
shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())


def is_content_word(word: str) -> bool:
    return word.isalpha() and word.lower() not in STOP_WORDS


def content_token_table(tokenizer) -> torch.Tensor:
    """(vocab,) bool: True for the token ids whose word piece (CLIP BPE, "</w>" marks a word end) is a content word."""
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return torch.tensor([is_content_word(p.replace("</w>", "")) for p in pieces], dtype=torch.bool)
```

In `mmae/models/masking.py` change `random_token_mask` to:

```python
def random_token_mask(
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator | None = None,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask `max(1, round(ratio * n))` tokens of each caption, n = its real, non-special tokens.

    With `allowed` (B, T) bool, the masked tokens are drawn from the allowed ones only (the count is still set by
    n, capped by the number allowed). Captions with no maskable token get no mask. Returns (B, T) bool.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"token mask ratio must be in (0, 1), got {ratio}")
    real = attention_mask.bool() & ~special_tokens_mask.bool()
    maskable = real if allowed is None else real & allowed.bool()
    n = real.sum(dim=1)
    available = maskable.sum(dim=1)
    k = torch.clamp(torch.round(n.float() * ratio), min=1).long()
    k = torch.where(available > 0, torch.minimum(k, available), torch.zeros_like(k))
    noise = _uniform(tuple(maskable.shape), maskable.device, generator)
    noise = noise.masked_fill(~maskable, 2.0)  # non-maskable positions sort after every maskable one
    ranks = noise.argsort(dim=1).argsort(dim=1)
    return ranks < k.unsqueeze(1)
```

(With `allowed=None`, `available == n`, so the result is identical to before.)

In `mmae/data/collate.py`:

```python
from mmae.data.stopwords import content_token_table


class Collator:
    def __init__(self, tokenizer_name: str, max_text_len: int, content_words: bool = False) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = int(max_text_len)
        self.content_table = content_token_table(self.tokenizer) if content_words else None

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
        out = {key: enc[key] for key in ("input_ids", "attention_mask", "special_tokens_mask")}
        if self.content_table is not None:
            out["content_tokens_mask"] = (
                self.content_table[out["input_ids"]] & out["attention_mask"].bool() & ~out["special_tokens_mask"].bool()
            )
        return out
```

(`pairs` and `retrieval` stay as they are; `retrieval` reshapes every key.)

In `mmae/models/model.py` `__init__`, after the M1 block:

```python
        self.text_mode = str(cfg.masking.get("text_mode", "random"))
        if self.text_mode not in ("random", "content"):
            raise ValueError(f"masking.text_mode must be 'random' or 'content', got {self.text_mode!r}")
```

and in `forward` replace the `token_mask = random_token_mask(...)` line with:

```python
            allowed = None
            if self.text_mode == "content":
                if "content_tokens_mask" not in batch:
                    raise KeyError(
                        "masking.text_mode=content needs batch['content_tokens_mask']: build the Collator with "
                        "content_words=True"
                    )
                allowed = batch["content_tokens_mask"]
            token_mask = random_token_mask(attention_mask, batch["special_tokens_mask"], self.text_ratio, allowed=allowed)
```

In `mmae/engine/trainer.py` `Trainer.__init__`, replace `self.collator = Collator(processor, dcfg.max_text_len)` with:

```python
        content_words = str(mcfg.masking.get("text_mode", "random")) == "content"
        self.collator = Collator(processor, dcfg.max_text_len, content_words=content_words)
```

- [ ] **Step 5: Run** `python -m pytest -q tests/test_masking.py tests/test_data.py tests/test_model_variants.py tests/test_model.py` → PASS. Break the guard once (pass `allowed=None` in the model's content branch) and confirm `test_m2b_masks_only_content_words` fails; restore.

- [ ] **Step 6: End-to-end smoke** — `python -m pytest -q tests/test_train_smoke.py` → PASS; then on the fake COCO via a one-off Python snippet or the existing `run_train` helper, run `train.py model=fusion_multilearner model.masking.text_mode=content` once and confirm exit code 0 (paste the last log line in the report).

- [ ] **Step 7: Full suite, change log, commit** `feat(masking): M2b content-word masking switch`.

---

### Task 4: M3, decoders read the other modality's clean pooled embedding (`model.pooled_conditioning`)

**Files:**
- Modify: `configs/model/base.yaml`, `mmae/models/fusion.py` (add `append_memory_token`), `mmae/models/model.py`
- Test: `tests/test_fusion_decoders.py` (append), `tests/test_model_variants.py` (append)

**Interfaces:**
- Consumes: Task 2's `image_emb`, `text_emb`, `fused`, `text_fused` in `forward`.
- Produces: `append_memory_token(memory: Tensor, padding: Tensor | None, token: Tensor) -> tuple[Tensor, Tensor]` in `mmae/models/fusion.py`; `MultiMAE.pooled_conditioning: bool`; modules `pooled_image_proj`, `pooled_text_proj` (`nn.Linear(embed_dim, fusion.dim)`) only when on.

- [ ] **Step 1: Config key** in `configs/model/base.yaml` after `mlm_image_source`:

```yaml
pooled_conditioning: false   # M3: true = the text decoder also reads the clean image embedding and the image
                             # decoder the clean text embedding (one extra memory token each; never their own)
```

- [ ] **Step 2: Failing tests.** Append to `tests/test_fusion_decoders.py`:

```python
def test_append_memory_token_adds_one_visible_position():
    from mmae.models.fusion import append_memory_token

    memory = torch.randn(2, 5, 8)
    token = torch.randn(2, 8)
    out, padding = append_memory_token(memory, None, token)
    assert out.shape == (2, 6, 8) and torch.equal(out[:, -1], token) and torch.equal(out[:, :5], memory)
    assert padding.shape == (2, 6) and not padding.any()
    given = torch.tensor([[False] * 4 + [True], [False] * 5])
    _, padding = append_memory_token(memory, given, token)
    assert torch.equal(padding[:, :5], given) and not padding[:, 5].any()
```

In `tests/test_model_variants.py` add `"m3_pooled": ("model.pooled_conditioning=true",),` to `VARIANTS` and append:

```python
def test_m3_each_decoder_reads_the_other_modality(tokenizer, monkeypatch):
    model = tiny_model("fusion_multilearner", "model.pooled_conditioning=true").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[0]
    assert not torch.equal(run(change(batch, model, masks, "image", True))[0]["text_decoder"], base["text_decoder"])
    assert not torch.equal(run(change(batch, model, masks, "text", True))[0]["image_decoder"], base["image_decoder"])


@pytest.mark.parametrize("name", ["fusion_none", "fusion_concat"])
def test_m3_works_with_other_fusions(name, tokenizer):
    model = tiny_model(name, "model.pooled_conditioning=true")
    model(make_batch(tokenizer))["loss"].backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing


@pytest.mark.parametrize("name", ["contrastive", "image_mae", "text_mlm"])
def test_m3_needs_both_modalities_and_reconstruction(name):
    with pytest.raises(ValueError, match="pooled_conditioning"):
        tiny_model(name, "model.pooled_conditioning=true")
```

Extend `test_model_builds_from_a_config_without_the_new_keys` with `assert model.pooled_conditioning is False`.

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Implement.** In `mmae/models/fusion.py` add (after `concat_modalities`):

```python
def append_memory_token(
    memory: torch.Tensor, padding: torch.Tensor | None, token: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one always-visible token (B, dim) to a decoder memory (B, L, dim); padding (B, L) may be None."""
    if padding is None:
        padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
    memory = torch.cat([memory, token.unsqueeze(1).to(memory.dtype)], dim=1)
    padding = torch.cat([padding, torch.zeros_like(padding[:, :1])], dim=1)
    return memory, padding
```

In `mmae/models/model.py`: import `append_memory_token` from `mmae.models.fusion`. In `__init__`, after `self.fusion = build_fusion(cfg.fusion)` (inside `if self.reconstruction:`):

```python
        self.pooled_conditioning = bool(cfg.get("pooled_conditioning", False))
        if self.pooled_conditioning:
            if not (self.reconstruction and self.has_contrastive):
                raise ValueError("pooled_conditioning needs both modalities and reconstruction")
            embed_dim = towers.vision.projection.out_features
            self.pooled_image_proj = nn.Linear(embed_dim, dim)  # clean image embedding -> text-decoder memory token
            self.pooled_text_proj = nn.Linear(embed_dim, dim)   # clean text embedding -> image-decoder memory token
```

Place this block so it runs for every model (also `reconstruction: false`, where it must raise when on): put it after the `if self.reconstruction: self.fusion = ...` lines, outside that `if`.

In `forward`, replace the two decoder blocks with:

```python
        image_memory, image_padding = fused.image_memory, fused.image_padding
        text_memory, text_memory_padding = text_fused.text_memory, text_fused.text_padding
        if self.pooled_conditioning:  # M3: each decoder also reads the OTHER modality's clean pooled embedding
            if self.use_image:
                image_memory, image_padding = append_memory_token(image_memory, image_padding, self.pooled_text_proj(text_emb))
            if self.use_text:
                text_memory, text_memory_padding = append_memory_token(
                    text_memory, text_memory_padding, self.pooled_image_proj(image_emb)
                )
        if self.use_image:
            pred = self.image_decoder(image_memory, image_padding)
            losses["mae"] = mae_loss(pred, images, patch_mask, self.vision.patch_size, norm_pix=self.norm_pix)
        if self.use_text:
            logits = self.text_decoder(text_memory, text_memory_padding, query_padding=text_padding)
            losses["mlm"] = mlm_loss(logits, input_ids, token_mask)
```

- [ ] **Step 5: Run** `python -m pytest -q tests/test_fusion_decoders.py tests/test_model_variants.py tests/test_model.py` → PASS. Break the guard once (feed `self.pooled_image_proj(text_emb)` to the text decoder, i.e. the text decoder's own clean embedding) and confirm `test_each_decoder_ignores_its_own_masked_content[m3_pooled]` fails; restore.

- [ ] **Step 6: Full suite, change log, commit** `feat(model): M3 cross-modal pooled conditioning switch`.

---

### Task 5: M6, masked-view InfoNCE (`model.loss.weights.masked_view`)

**Files:**
- Modify: `configs/model/base.yaml`, `mmae/models/model.py`
- Test: `tests/test_model_variants.py` (append)

**Interfaces:**
- Consumes: Task 2's `masked_text_hidden`, `image_emb`; Task 1's `text.pool`.
- Produces: output key `loss_masked_view` (only when the weight > 0); `MultiMAE.masked_view_weight: float`.

- [ ] **Step 1: Config key** under `loss.weights` in `configs/model/base.yaml`:

```yaml
    masked_view: 0.0       # M6: InfoNCE of the masked caption's pooled embedding against the clean images
```

- [ ] **Step 2: Failing tests.** In `tests/test_model_variants.py` add `"m6_masked_view": ("model.loss.weights.masked_view=0.25",),` to `VARIANTS` and append:

```python
def test_m6_adds_a_weighted_masked_view_loss(tokenizer):
    model = tiny_model("fusion_multilearner", "model.loss.weights.masked_view=0.25")
    out = model(make_batch(tokenizer))
    assert set(out) == {"loss", "loss_contrastive", "loss_mae", "loss_mlm", "loss_masked_view"}
    expected = out["loss_contrastive"] + out["loss_mae"] + out["loss_mlm"] + 0.25 * out["loss_masked_view"]
    assert torch.allclose(out["loss"], expected)
    assert "loss_masked_view" not in tiny_model("fusion_multilearner")(make_batch(tokenizer))


def test_m6_scores_the_masked_caption(tokenizer, monkeypatch):
    """The loss uses the masked view: masked tokens' content does not move it, visible tokens' content does."""
    model = tiny_model("fusion_multilearner", "model.loss.weights.masked_view=0.25").eval()
    masks, run = recorder(model, monkeypatch)
    batch = make_batch(tokenizer)
    base = run(batch)[1]["loss_masked_view"]
    assert torch.equal(run(change(batch, model, masks, "text", True))[1]["loss_masked_view"], base)
    assert not torch.equal(run(change(batch, model, masks, "text", False))[1]["loss_masked_view"], base)


@pytest.mark.parametrize("name", ["contrastive", "image_mae", "text_mlm"])
def test_m6_needs_both_modalities_and_reconstruction(name):
    with pytest.raises(ValueError, match="masked_view"):
        tiny_model(name, "model.loss.weights.masked_view=0.25")
```

Extend `test_model_builds_from_a_config_without_the_new_keys` with `assert model.masked_view_weight == 0.0`.

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Implement.** In `__init__` after `self.loss_weights = ...`:

```python
        self.masked_view_weight = float(cfg.loss.weights.get("masked_view", 0.0))
```

and after the M1 validation:

```python
        if self.masked_view_weight > 0 and not (self.reconstruction and self.has_contrastive):
            raise ValueError("loss.weights.masked_view > 0 needs both modalities and reconstruction")
```

In `forward`, right after `text_fused` is set (before the decoders):

```python
        if self.masked_view_weight > 0:  # M6: the masked caption is a less specific view of the same image
            masked_emb = self.text.pool(masked_text_hidden, input_ids, attention_mask)
            losses["masked_view"] = contrastive_loss(image_emb, masked_emb, self.logit_scale, self.gather)
```

(`out["loss"]` already sums `self.loss_weights[name] * value`; `masked_view` is in `loss_weights` from the config.)

- [ ] **Step 5: Run** `python -m pytest -q tests/test_model_variants.py tests/test_model.py` → PASS. Break the guard once (pool the clean `self.text.encode(input_ids, attention_mask)` instead of `masked_text_hidden`) and confirm `test_m6_scores_the_masked_caption` fails; restore.

- [ ] **Step 6: Full suite, change log, commit** `feat(model): M6 masked-view InfoNCE switch`.

---

### Task 6: R2, per-tower learning rates, layer-wise decay and a frozen vision warmup

**Files:**
- Modify: `configs/train/default.yaml`, `configs/train/debug.yaml`, `mmae/models/model.py` (`param_groups`, new `tower_layer_id`), `mmae/engine/trainer.py` (optimizer and scheduler construction)
- Test: `tests/test_model.py` (append), `tests/test_trainer.py` (append)

**Interfaces:**
- Produces: `tower_layer_id(name: str, num_layers: int) -> int` in `mmae/models/model.py`; `MultiMAE.param_groups(lr, lr_backbone, weight_decay, lr_text=None, lr_vision=None, layer_decay=1.0, split_towers=False) -> list[dict]` where split groups carry `"tower": "vision" | "text" | None`; train keys `lr_text`, `lr_vision`, `layer_decay`, `freeze_vision_epochs`.

- [ ] **Step 1: Config keys.** Append to the `train` keys of BOTH `configs/train/default.yaml` and `configs/train/debug.yaml` (debug under its `train:` block, same indentation as its other keys):

```yaml
lr_text: null            # CLIP text tower lr; null = lr_backbone (R2 uses 5.0e-5)
lr_vision: null          # CLIP vision tower lr; null = lr_backbone (R2 uses 5.0e-6)
layer_decay: 1.0         # layer-wise lr decay inside each tower, BEiT-style (R2 uses 0.7); 1.0 = none
freeze_vision_epochs: 0  # the vision tower's lr is 0 for the first N epochs (R2 uses 2)
```

- [ ] **Step 2: Failing tests.** Append to `tests/test_model.py`:

```python
def test_tower_layer_id():
    from mmae.models.model import tower_layer_id

    assert tower_layer_id("model.embeddings.patch_embedding.weight", 12) == 0
    assert tower_layer_id("model.pre_layrnorm.weight", 12) == 0
    assert tower_layer_id("model.encoder.layers.0.mlp.fc1.weight", 12) == 1
    assert tower_layer_id("model.encoder.layers.11.mlp.fc1.weight", 12) == 12
    assert tower_layer_id("model.post_layernorm.weight", 12) == 13
    assert tower_layer_id("projection.weight", 12) == 13


def test_split_param_groups_per_tower_and_layer():
    model = tiny_model("fusion_concat")
    groups = model.param_groups(1e-4, 1e-5, 0.05, lr_text=5e-5, lr_vision=5e-6, layer_decay=0.5)
    seen = [id(p) for g in groups for p in g["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable) and len(seen) == len(set(seen))

    def lr_of(param):
        return next(g["lr"] for g in groups if any(p is param for p in g["params"]))

    layers = len(model.vision.model.encoder.layers)  # 2 in the tiny CLIP
    assert lr_of(model.text.projection.weight) == pytest.approx(5e-5)                    # top: no decay
    assert lr_of(model.vision.projection.weight) == pytest.approx(5e-6)
    assert lr_of(model.vision.model.encoder.layers[0].mlp.fc1.weight) == pytest.approx(5e-6 * 0.5 ** layers)
    assert lr_of(model.vision.model.embeddings.patch_embedding.weight) == pytest.approx(5e-6 * 0.5 ** (layers + 1))
    assert lr_of(model.image_proj.weight) == pytest.approx(1e-4)                          # new modules
    assert lr_of(model.text.mask_embedding) == pytest.approx(1e-4)
    towers = {g["tower"] for g in groups if any(p is model.vision.projection.weight for p in g["params"])}
    assert towers == {"vision"}
    assert all(g["name"].startswith("backbone") == (g["tower"] is not None) for g in groups)
```

Append to `tests/test_trainer.py`:

```python
def _snapshot(params):
    return [p.detach().clone() for p in params]


def _same(params, snapshot):
    return all(torch.equal(p, s) for p, s in zip(params, snapshot))


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_frozen_vision_warmup(tmp_path, fake_coco, accelerator, grad_accum):
    cfg = tiny_cfg(tmp_path, fake_coco, "model=contrastive", "train.epochs=2", "train.freeze_vision_epochs=1",
                   f"train.grad_accum={grad_accum}")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run, None))
        model = accelerator.unwrap_model(trainer.model)
        vision, text = list(model.vision.pretrained_parameters()), list(model.text.pretrained_parameters())
        v0, t0 = _snapshot(vision), _snapshot(text)
        trainer.train_epoch(1)
        assert _same(vision, v0), "vision tower moved during the frozen epoch"
        assert not _same(text, t0), "text tower did not train"
        trainer.train_epoch(2)
        assert not _same(vision, v0), "vision tower did not train after the frozen epoch"


def test_default_schedule_keeps_the_legacy_groups(tmp_path, fake_coco, accelerator):
    cfg = tiny_cfg(tmp_path, fake_coco, "model=fusion_concat")
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run, None))
        names = sorted(g["name"] for g in trainer.optimizer.param_groups)
    assert names == ["backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"]
```

(Check how existing tests in `tests/test_trainer.py` construct `Run` and `Trainer` — e.g. `test_fit_restores_best_weights_before_test` — and mirror that exactly if it differs from the above; `Run(cfg, enabled=False)` and `MetricLogger(run, None)` are the intended non-writing forms.)

- [ ] **Step 3: Run, expect FAIL.**

- [ ] **Step 4: Implement** in `mmae/models/model.py` (add `import re` at the top):

```python
def tower_layer_id(name: str, num_layers: int) -> int:
    """BEiT-style depth index of a tower parameter (name relative to the tower): 0 for the embeddings and the
    vision pre-norm, i + 1 for transformer layer i, num_layers + 1 for the final norm and the projection."""
    match = re.search(r"encoder\.layers\.(\d+)\.", name)
    if match:
        return int(match.group(1)) + 1
    if "embeddings." in name or "pre_layrnorm" in name:
        return 0
    return num_layers + 1
```

Replace `param_groups` with:

```python
    def param_groups(
        self, lr: float, lr_backbone: float, weight_decay: float, lr_text: float | None = None,
        lr_vision: float | None = None, layer_decay: float = 1.0, split_towers: bool = False,
    ) -> list[dict]:
        """AdamW groups: pretrained tower weights at the backbone lr, everything else at lr; no decay on biases,
        norms, embeddings, queries and the logit scale. With per-tower lrs, layer decay or split_towers, the
        tower weights are grouped per tower and layer (lr * layer_decay ** (num_layers + 1 - layer)), and every
        group carries "tower" ("vision", "text" or None) for the trainer's frozen warmup."""
        split = split_towers or lr_text is not None or lr_vision is not None or layer_decay != 1.0
        backbone = {id(p) for tower in self.towers() for p in tower.pretrained_parameters()}
        if not split:
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

        tower_lr = {"vision": lr_backbone if lr_vision is None else lr_vision,
                    "text": lr_backbone if lr_text is None else lr_text}
        place: dict[int, tuple[str, int, int]] = {}  # id(param) -> (tower, layer, num_layers)
        for tower_name, tower in (("vision", self.vision), ("text", self.text)):
            if tower is None:
                continue
            num_layers = len(tower.model.encoder.layers)
            for name, param in tower.named_parameters():
                if id(param) in backbone:
                    place[id(param)] = (tower_name, tower_layer_id(name, num_layers), num_layers)
        split_groups: dict[tuple[str, int, bool], dict] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim < 2 or any(key in name for key in NO_DECAY_KEYS)
            suffix = "no_decay" if no_decay else "decay"
            if id(param) in place:
                tower_name, layer, num_layers = place[id(param)]
                key = (tower_name, layer, no_decay)
                group = split_groups.setdefault(key, {
                    "params": [], "lr": tower_lr[tower_name] * layer_decay ** (num_layers + 1 - layer),
                    "weight_decay": 0.0 if no_decay else weight_decay,
                    "name": f"backbone_{tower_name}_layer{layer:02d}_{suffix}", "tower": tower_name,
                })
            else:
                key = ("head", -1, no_decay)
                group = split_groups.setdefault(key, {
                    "params": [], "lr": lr, "weight_decay": 0.0 if no_decay else weight_decay,
                    "name": f"head_{suffix}", "tower": None,
                })
            group["params"].append(param)
        return [split_groups[key] for key in sorted(split_groups)]
```

In `mmae/engine/trainer.py` `Trainer.__init__`, replace the optimizer line and the scheduler block with:

```python
        freeze_epochs = int(tcfg.get("freeze_vision_epochs", 0))
        optimizer = torch.optim.AdamW(model.param_groups(
            tcfg.lr, tcfg.lr_backbone, tcfg.weight_decay, lr_text=tcfg.get("lr_text"),
            lr_vision=tcfg.get("lr_vision"), layer_decay=float(tcfg.get("layer_decay", 1.0)),
            split_towers=freeze_epochs > 0,
        ))
```

(keep `accelerator.prepare(...)` and the empty-loader check as they are), then:

```python
        steps_per_epoch = math.ceil(len(self.train_loader) / tcfg.grad_accum)
        total_steps = steps_per_epoch * tcfg.epochs

        def schedule(step: int) -> float:
            return warmup_cosine(step, tcfg.warmup_steps, total_steps)

        lambdas = schedule
        if freeze_epochs > 0:  # R2: vision lr 0 for the first freeze_epochs epochs, counted in optimizer steps
            freeze_steps = steps_per_epoch * freeze_epochs

            def frozen(step: int) -> float:
                return 0.0 if step < freeze_steps else schedule(step)

            lambdas = [frozen if group.get("tower") == "vision" else schedule for group in optimizer.param_groups]
        # Built on the raw optimizer and stepped by hand once per optimizer step, so it does not depend on
        # accelerate's scheduler stepping rules.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
```

(AdamW with lr 0 leaves a parameter bit-identical: both the decoupled weight decay and the update are scaled by lr. Gradient clipping still sees the frozen tower's gradients; that matches computing them anyway.)

- [ ] **Step 5: Run** `python -m pytest -q tests/test_model.py tests/test_trainer.py` → PASS. Break the guard once (count `freeze_steps` in batches: `len(self.train_loader) * freeze_epochs`) and confirm `test_frozen_vision_warmup[2]` fails; restore.

- [ ] **Step 6: Full suite, change log, commit** `feat(train): R2 per-tower lrs, layer decay, frozen vision warmup`.

---

### Task 7: Seeded training sampler (`train.seeded_sampler`)

**Files:**
- Modify: `configs/train/default.yaml`, `configs/train/debug.yaml`, `mmae/engine/trainer.py` (`_loader`, train loader construction)
- Test: `tests/test_trainer.py` (append)

**Interfaces:**
- Produces: train key `seeded_sampler: false`; `Trainer._loader(..., generator: torch.Generator | None = None)`.

- [ ] **Step 1: Config key** in BOTH train configs:

```yaml
seeded_sampler: false    # true: the training shuffle uses a generator seeded by cfg.seed alone, so seed k gives
                         # every model type the same data order (Stage 2's paired comparisons)
```

- [ ] **Step 2: Failing test** (append to `tests/test_trainer.py`):

```python
def _first_epoch_order(tmp_path, fake_coco, accelerator, *overrides):
    cfg = tiny_cfg(tmp_path, fake_coco, *overrides)
    torch.manual_seed(cfg.seed)  # as train.py's set_seed
    with Run(cfg, enabled=False) as run:
        trainer = Trainer(cfg, accelerator, run, MetricLogger(run, None))
        return torch.cat([batch["input_ids"] for batch in trainer.train_loader])


def test_seeded_sampler_gives_every_model_the_same_order(tmp_path, fake_coco, accelerator):
    on = ("train.seeded_sampler=true", "train.batch_size=2")
    a = _first_epoch_order(tmp_path / "a", fake_coco, accelerator, "model=contrastive", *on)
    b = _first_epoch_order(tmp_path / "b", fake_coco, accelerator, "model=fusion_multilearner", *on)
    c = _first_epoch_order(tmp_path / "c", fake_coco, accelerator, "model=contrastive", *on, "seed=7")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
```

- [ ] **Step 3: Run, expect FAIL** (`a` and `b` differ: model construction consumes the global RNG differently).

- [ ] **Step 4: Implement.** In `Trainer._loader` add a `generator: torch.Generator | None = None` parameter and pass `generator=generator` to `DataLoader`. In `__init__`:

```python
        generator = torch.Generator().manual_seed(int(cfg.seed)) if tcfg.get("seeded_sampler", False) else None
        train_loader = self._loader(train_set, tcfg.batch_size, shuffle=True, drop_last=True,
                                    collate=self.collator.pairs, generator=generator)
```

- [ ] **Step 5: Run** `python -m pytest -q tests/test_trainer.py` → PASS (if `a != b` still, check whether accelerate replaced the sampler's generator in `prepare`; the fix is to keep our generator, not to change the test). Break the guard once (drop `generator=generator` from the DataLoader) and confirm the test fails; restore.

- [ ] **Step 6: Full suite, change log, commit** `feat(train): seeded training sampler switch`.

---

### Task 8: VWSD evaluation (`mmae/engine/vwsd.py`, `evaluate.py eval.vwsd_dir`)

**Files:**
- Create: `mmae/engine/vwsd.py`, `tests/test_vwsd.py`
- Modify: `configs/config.yaml` (`eval` keys), `evaluate.py`, `tests/helpers.py` (`make_fake_vwsd`)

**Interfaces:**
- Produces: `VwsdItem(word, phrase, candidates, gold)`, `read_vwsd(root, lang="en") -> list[VwsdItem]`, `vwsd_metrics(image_emb, text_emb, items, index) -> dict[str, float]` with keys `vwsd/hit1`, `vwsd/mrr` (percent) and `vwsd/n`, `evaluate_vwsd(model, root, transform, collator, lang="en", prompt="{phrase}", batch_size=256, num_workers=0, device=None) -> dict[str, float]`; config `eval.vwsd_dir: null`, `eval.vwsd_lang: en`, `eval.vwsd_prompt: "{phrase}"`; `helpers.make_fake_vwsd(root) -> Path`.

Data facts (checked on the released package at `/data/SSD/vwsd/`): files `en.test.data.v1.1.txt` (463 lines: `target<TAB>phrase<TAB>10 image names`), `en.test.gold.v1.1.txt` (463 gold names), `fa.test.data.txt` / `fa.test.gold.txt`, `it.test.*.v1.1.txt`, images in `test_images_resized/` (8,100 files, `.jpg/.JPG/.png/.PNG/.jpeg`; names in the data files match the disk exactly). License CC-BY-NC 4.0.

- [ ] **Step 1: Config keys** in `configs/config.yaml` under `eval:`:

```yaml
  vwsd_dir: null           # SemEval-2023 VWSD test package (local /data/SSD/vwsd, node /local/wding/Dataset/vwsd); null skips
  vwsd_lang: en
  vwsd_prompt: "{phrase}"  # text query; {phrase} and {word} are filled in
```

- [ ] **Step 2: Failing tests.** Add to `tests/helpers.py`:

```python
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
```

Create `tests/test_vwsd.py`:

```python
import json

import pytest
import torch

from helpers import CLIP_NAME, make_fake_vwsd, run_train
from mmae.engine.vwsd import VwsdItem, evaluate_vwsd, read_vwsd, vwsd_metrics


def test_read_vwsd(tmp_path):
    items = read_vwsd(make_fake_vwsd(tmp_path / "vwsd"))
    assert len(items) == 3
    assert items[0] == VwsdItem("goal", "football goal", tuple(f"image.{i}." + ("png" if i % 3 == 1 else "jpg") for i in range(10)), "image.3.jpg")


def test_read_vwsd_rejects_a_gold_outside_the_candidates(tmp_path):
    root = make_fake_vwsd(tmp_path / "vwsd")
    (root / "en.test.gold.v1.1.txt").write_text("image.11.jpg\nimage.11.jpg\nimage.1.png\n")
    with pytest.raises(ValueError, match="line 1"):
        read_vwsd(root)


def test_read_vwsd_rejects_mismatched_files(tmp_path):
    root = make_fake_vwsd(tmp_path / "vwsd")
    (root / "en.test.gold.v1.1.txt").write_text("image.3.jpg\n")
    with pytest.raises(ValueError, match="3 items but 1 gold"):
        read_vwsd(root)


def test_vwsd_metrics_ranks():
    items = [VwsdItem("w", "p", ("a", "b", "c"), "a"), VwsdItem("w", "p", ("a", "b", "c"), "b")]
    index = {"a": 0, "b": 1, "c": 2}
    image_emb = torch.eye(3)
    text_emb = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])  # gold "a" ranks 1st, gold "b" 2nd
    metrics = vwsd_metrics(image_emb, text_emb, items, index)
    assert metrics == {"vwsd/hit1": 50.0, "vwsd/mrr": 75.0, "vwsd/n": 2.0}


def test_evaluate_vwsd_on_the_tiny_model(tmp_path):
    from helpers import compose_cfg
    from mmae.data import Collator, build_image_transform
    from mmae.models import MultiMAE

    cfg = compose_cfg("model=contrastive", "model.backbone.pretrained=tiny-random-clip")
    model = MultiMAE(cfg.model, max_text_len=32)
    metrics = evaluate_vwsd(model, make_fake_vwsd(tmp_path / "vwsd"), build_image_transform(CLIP_NAME),
                            Collator(CLIP_NAME, 32), batch_size=4)
    assert metrics["vwsd/n"] == 3.0
    assert 0.0 <= metrics["vwsd/hit1"] <= 100.0 and 10.0 <= metrics["vwsd/mrr"] <= 100.0


def test_evaluate_py_adds_vwsd(tmp_path, fake_coco):
    vwsd = make_fake_vwsd(tmp_path / "vwsd")
    out = tmp_path / "metrics.json"
    result = run_train(tmp_path, fake_coco, "model=contrastive", f"eval.vwsd_dir={vwsd}", f"eval.output={out}",
                       script="evaluate.py")
    assert result.returncode == 0, result.stderr[-3000:]
    metrics = json.loads(out.read_text())
    assert metrics["vwsd/n"] == 3.0 and "rsum" in metrics
```

- [ ] **Step 3: Run, expect FAIL** (`ModuleNotFoundError: mmae.engine.vwsd`).

- [ ] **Step 4: Implement** `mmae/engine/vwsd.py`:

```python
"""SemEval-2023 Task 1, Visual Word Sense Disambiguation (VWSD; Raganato et al., 2023; data CC-BY-NC 4.0).

Each item is a possibly ambiguous target word, a short phrase that fixes its sense ("football goal") and ten
candidate images. The model ranks the candidates by cosine similarity between the phrase's text embedding and the
image embeddings; Hit@1 and MRR are reported in percent (rank = 1 + the number of candidates scored strictly
higher, as in mmae.engine.retrieval). Layout of `root`, the released test package: {lang}.test.data*.txt
(target <tab> phrase <tab> 10 image names), {lang}.test.gold*.txt (the gold image name per line) and
test_images_resized/ with the images.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGES_SUBDIR = "test_images_resized"


@dataclass(frozen=True)
class VwsdItem:
    word: str
    phrase: str
    candidates: tuple[str, ...]
    gold: str


def _one_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {root}, found {[m.name for m in matches]}")
    return matches[0]


def read_vwsd(root: str | Path, lang: str = "en") -> list[VwsdItem]:
    root = Path(root)
    lines = [l for l in _one_file(root, f"{lang}.test.data*.txt").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    gold = [l.strip() for l in _one_file(root, f"{lang}.test.gold*.txt").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if len(lines) != len(gold):
        raise ValueError(f"{lang} VWSD: {len(lines)} items but {len(gold)} gold labels")
    items = []
    for number, (line, answer) in enumerate(zip(lines, gold), start=1):
        fields = [f.strip() for f in line.split("\t")]
        candidates = tuple(f for f in fields[2:] if f)
        if answer not in candidates:
            raise ValueError(f"{lang} VWSD line {number}: gold image {answer!r} is not one of its candidates")
        items.append(VwsdItem(fields[0], fields[1], candidates, answer))
    return items


class VwsdImages(Dataset):
    def __init__(self, root: str | Path, names: list[str], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.folder = Path(root) / IMAGES_SUBDIR
        self.names = names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.folder / self.names[i]) as image:
            return self.transform(image.convert("RGB"))  # PNGs with alpha or palettes, grayscale JPEGs


def vwsd_metrics(image_emb: torch.Tensor, text_emb: torch.Tensor, items: list[VwsdItem], index: dict[str, int]) -> dict[str, float]:
    """image_emb (M, D) rows indexed by image name (index), text_emb (Q, D) one row per item."""
    ranks = []
    for row, item in enumerate(items):
        scores = image_emb[[index[name] for name in item.candidates]] @ text_emb[row]
        gold = scores[item.candidates.index(item.gold)]
        ranks.append(1 + int((scores > gold).sum()))
    ranks_t = torch.tensor(ranks, dtype=torch.double)
    return {
        "vwsd/hit1": 100.0 * (ranks_t == 1).double().mean().item(),
        "vwsd/mrr": 100.0 * (1.0 / ranks_t).mean().item(),
        "vwsd/n": float(len(items)),
    }


@torch.no_grad()
def evaluate_vwsd(
    model, root: str | Path, transform, collator, lang: str = "en", prompt: str = "{phrase}",
    batch_size: int = 256, num_workers: int = 0, device: torch.device | str | None = None,
) -> dict[str, float]:
    """Hit@1 and MRR of `model` (an unwrapped MultiMAE: embed_image / embed_text) on the VWSD test set."""
    items = read_vwsd(root, lang)
    names = sorted({name for item in items for name in item.candidates})
    index = {name: i for i, name in enumerate(names)}
    device = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    loader = DataLoader(VwsdImages(root, names, transform), batch_size=batch_size, num_workers=num_workers)
    image_emb = torch.cat([model.embed_image(images.to(device)).float() for images in loader])
    queries = [prompt.format(phrase=item.phrase, word=item.word) for item in items]
    text_emb = []
    for start in range(0, len(queries), batch_size):
        tokens = collator.tokenize(queries[start : start + batch_size])
        text_emb.append(model.embed_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device)).float())
    model.train(was_training)
    return vwsd_metrics(image_emb.cpu(), torch.cat(text_emb).cpu(), items, index)
```

In `evaluate.py`: import `from mmae.engine.vwsd import evaluate_vwsd`; inside `if accelerator.is_main_process:` right after `metrics.update(extended(...))`:

```python
        if cfg.eval.get("vwsd_dir"):
            metrics.update(evaluate_vwsd(
                accelerator.unwrap_model(model), cfg.eval.vwsd_dir, build_image_transform(processor),
                Collator(processor, max_text_len), lang=cfg.eval.vwsd_lang, prompt=cfg.eval.vwsd_prompt,
                batch_size=cfg.train.eval_batch_size, num_workers=cfg.train.num_workers, device=accelerator.device,
            ))
```

- [ ] **Step 5: Run** `python -m pytest -q tests/test_vwsd.py tests/test_evaluate.py` → PASS.

- [ ] **Step 6: Real-data check on CPU** (no GPU): zero-shot B/32 on the real package:
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python -c "from omegaconf import OmegaConf; ..."` is awkward; instead run
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python evaluate.py model=contrastive data.limit_test=50 eval.extended_metrics=false eval.vwsd_dir=/data/SSD/vwsd eval.output=<scratch>/vwsd_zeroshot.json wandb.enabled=false train.num_workers=4`
and report `vwsd/hit1` and `vwsd/mrr` (expect well above chance, Hit@1 10%; published CLIP baselines are around 60). This takes several minutes on CPU.

- [ ] **Step 7: Full suite, change log, commit** `feat(eval): VWSD Hit@1/MRR in evaluate.py`.

---

### Task 9: Stage 0 diagnostics library (`mmae/engine/diagnostics.py`)

**Files:**
- Create: `mmae/engine/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `mmae.engine.eccv.pmrp_ground_truth`, `mmae.data.stopwords.is_content_word`.
- Produces:
  - `class_set_groups(pm_gt: dict, image_ids: np.ndarray, caption_ids: np.ndarray) -> np.ndarray` (N,) int group per test image, -1 without PM entry.
  - `neighbour_purity(emb: Tensor, groups: np.ndarray, k: int = 10, owners: np.ndarray | None = None, chunk: int = 1024) -> float` (percent).
  - `pmrp_rows(pm_gt, image_ids, caption_ids) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]` (direction -> query rows, positive rows; caption rows are flat `image_row * K + j`).
  - `per_query_rprecision(query_emb, cand_emb, positives: list[np.ndarray], max_r: int = 50, chunk: int = 1024) -> np.ndarray` (fractions).
  - `similarity_stats(image_emb, caption_emb, groups, chunk: int = 512) -> dict[str, float]` (`pos`, `same_class_neg`, `other_neg` mean cosines).
  - `COCO_CLASS_WORDS: frozenset[str]`, `count_class_words(caption: str) -> int`.
  - `drop_words(caption: str, kind: str, k: int, rng: random.Random) -> str | None`.

- [ ] **Step 1: Failing tests** — create `tests/test_diagnostics.py`:

```python
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mmae.engine.diagnostics import (
    class_set_groups, count_class_words, drop_words, neighbour_purity, per_query_rprecision, pmrp_rows,
    similarity_stats,
)

# 4 test images (COCO ids 10, 11, 12, 13), 2 captions each (ids 100..107); images 10 and 11 share a class set,
# 12 is alone, 13 has no PM entry.
IMAGE_IDS = np.array([10, 11, 12, 13])
CAPTION_IDS = np.array([[100, 101], [102, 103], [104, 105], [106, 107]])
PM = {"t2i": {100: [11], 101: [11], 102: [10], 103: [10], 104: [], 105: []}, "i2t": {10: [102, 103], 11: [100, 101]}}


def test_class_set_groups():
    groups = class_set_groups(PM, IMAGE_IDS, CAPTION_IDS)
    assert groups[0] == groups[1] and groups[2] not in (groups[0], -1) and groups[3] == -1


def test_neighbour_purity():
    # nearest other item: 0 -> 1, 1 -> 0, 2 -> 3; item 3 is alone in group 1 (no mate), so it is not a query
    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.4]]), dim=-1)
    groups = np.array([0, 0, 0, 1])
    assert neighbour_purity(emb, groups, k=1) == pytest.approx(200.0 / 3)
    # owners: items 0 and 1 share an owner, so neither may pick the other; both then pick item 3 (group 1)
    assert neighbour_purity(emb, groups, k=1, owners=np.array([0, 0, 1, 2])) == 0.0


def test_pmrp_rows_and_rprecision():
    rows = pmrp_rows(PM, IMAGE_IDS, CAPTION_IDS)
    queries, positives = rows["t2i"]
    assert queries.tolist() == [0, 1, 2, 3, 4, 5]
    assert sorted(positives[0].tolist()) == [0, 1]           # own image 10 (row 0) + image 11 (row 1)
    image_emb = torch.eye(4)
    # captions of images 10 and 11 rank their own image first and image 12 (row 2, not a positive) second;
    # captions of image 12 (one positive) rank it first
    caption_emb = torch.tensor([[1.0, 0, 0.5, 0], [1.0, 0, 0.5, 0], [0, 1.0, 0.5, 0], [0, 1.0, 0.5, 0],
                                [0, 0, 1.0, 0.5], [0, 0, 1.0, 0.5]])
    r = per_query_rprecision(caption_emb, image_emb, positives, max_r=50)
    assert r.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0, 1.0])


def test_similarity_stats():
    image_emb = torch.eye(3)
    caption_emb = torch.eye(3)[:, None, :].repeat(1, 2, 1)   # every caption equals its image
    stats = similarity_stats(image_emb, caption_emb, np.array([0, 0, 1]))
    assert stats == {"pos": 1.0, "same_class_neg": 0.0, "other_neg": 0.0}


def test_class_words_and_drop_words():
    assert count_class_words("A man riding a horse next to two dogs.") == 3  # man (person), horse, dogs
    rng = random.Random(0)
    assert drop_words("a dog on the beach", "content", 1, rng) in {"a on the beach", "a dog on the"}
    assert drop_words("a dog on the beach", "stop", 3, rng) == "dog beach"
    assert drop_words("a dog", "content", 2, rng) is None
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `mmae/engine/diagnostics.py`:

```python
"""Stage 0 diagnostics (spec 2026-10-03, section 5) on COCO 5k test embeddings.

Class sets come from the PMRP ground truth (pmrp_ground_truth, zeta = 0): a caption's t2i list holds every test
image whose COCO object-class set equals that of the caption's image, so images whose lists coincide share a set.
"""
from __future__ import annotations

import random
import re

import numpy as np
import torch

from mmae.data.stopwords import is_content_word
from mmae.engine.eccv import pmrp_ground_truth

# The 80 COCO category names as caption words (singular and plural), with common person words.
COCO_CLASS_WORDS = frozenset("""
person persons people man men woman women boy boys girl girls child children kid kids guy guys lady ladies player
players bicycle bicycles bike bikes car cars motorcycle motorcycles motorbike airplane airplanes plane planes jet bus
buses train trains truck trucks boat boats traffic light lights fire hydrant hydrants stop sign signs parking meter
meters bench benches bird birds cat cats dog dogs horse horses sheep cow cows elephant elephants bear bears zebra
zebras giraffe giraffes backpack backpacks umbrella umbrellas handbag handbags purse tie ties suitcase suitcases
frisbee frisbees skis ski snowboard snowboards ball balls kite kites bat bats glove gloves skateboard skateboards
surfboard surfboards racket rackets racquet bottle bottles glass glasses cup cups fork forks knife knives spoon
spoons bowl bowls banana bananas apple apples sandwich sandwiches orange oranges broccoli carrot carrots hotdog
pizza pizzas donut donuts doughnut doughnuts cake cakes chair chairs couch couches sofa plant plants bed beds table
tables toilet toilets tv tvs television laptop laptops computer mouse remote remotes keyboard keyboards phone phones
microwave microwaves oven ovens toaster sink sinks refrigerator refrigerators fridge book books clock clocks vase
vases scissors teddy toothbrush toothbrushes
""".split())
_WORD = re.compile(r"[A-Za-z']+")


def class_set_groups(pm_gt: dict, image_ids: np.ndarray, caption_ids: np.ndarray) -> np.ndarray:
    gt = pmrp_ground_truth(pm_gt, image_ids, caption_ids)
    owner = dict(zip(caption_ids.reshape(-1).tolist(), np.repeat(image_ids, caption_ids.shape[1]).tolist()))
    row = {image: i for i, image in enumerate(image_ids.tolist())}
    keys: dict[tuple[int, ...], int] = {}
    groups = np.full(len(image_ids), -1, dtype=np.int64)
    for caption, images in gt["t2i"].items():
        group = keys.setdefault(tuple(images), len(keys))
        groups[row[owner[caption]]] = group
    return groups


@torch.no_grad()
def neighbour_purity(emb: torch.Tensor, groups: np.ndarray, k: int = 10, owners: np.ndarray | None = None,
                     chunk: int = 1024) -> float:
    """Percent of each query's top-k most similar other items (cosine) in its own group, averaged over the queries
    with a group (>= 0) that has another member outside the query's owner. owners (n,): items with the query's
    owner are never neighbours (e.g. the other captions of a caption's image)."""
    n = emb.shape[0]
    g = torch.as_tensor(groups)
    own = torch.as_tensor(owners if owners is not None else np.arange(n))
    shares = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = emb[start:stop] @ emb.T
        scores[own[start:stop, None] == own[None, :]] = -float("inf")  # self and same-owner items
        top = scores.topk(k, dim=1).indices
        qg = g[start:stop]
        mates = ((g[None, :] == qg[:, None]) & (own[None, :] != own[start:stop, None])).sum(dim=1)
        valid = (qg >= 0) & (mates > 0)
        hit = (g[top] == qg[:, None]).double().mean(dim=1)
        shares.append(hit[valid])
    return 100.0 * torch.cat(shares).mean().item()


def pmrp_rows(pm_gt: dict, image_ids: np.ndarray, caption_ids: np.ndarray) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    gt = pmrp_ground_truth(pm_gt, image_ids, caption_ids)
    image_row = {image: i for i, image in enumerate(image_ids.tolist())}
    caption_row = {c: i for i, c in enumerate(caption_ids.reshape(-1).tolist())}
    t2i_q = sorted(gt["t2i"], key=caption_row.__getitem__)
    i2t_q = sorted(gt["i2t"], key=image_row.__getitem__)
    return {
        "t2i": (np.array([caption_row[c] for c in t2i_q]), [np.array([image_row[i] for i in gt["t2i"][c]]) for c in t2i_q]),
        "i2t": (np.array([image_row[i] for i in i2t_q]), [np.array([caption_row[c] for c in gt["i2t"][i]]) for i in i2t_q]),
    }


@torch.no_grad()
def per_query_rprecision(query_emb: torch.Tensor, cand_emb: torch.Tensor, positives: list[np.ndarray],
                         max_r: int = 50, chunk: int = 1024) -> np.ndarray:
    """R-Precision per query, R = min(#positives, max_r). Ties break by torch.topk, so the mean can differ from the
    package's PMRP (stable sort) by a few hundredths."""
    out = np.zeros(len(positives))
    depth = min(max_r, cand_emb.shape[0])
    for start in range(0, len(positives), chunk):
        stop = min(start + chunk, len(positives))
        top = (query_emb[start:stop] @ cand_emb.T).topk(depth, dim=1).indices.cpu().numpy()
        for j in range(stop - start):
            pos = positives[start + j]
            r = min(len(pos), max_r)
            out[start + j] = np.isin(top[j, :r], pos).sum() / r
    return out


@torch.no_grad()
def similarity_stats(image_emb: torch.Tensor, caption_emb: torch.Tensor, groups: np.ndarray, chunk: int = 512) -> dict[str, float]:
    """Mean image-caption cosine of own pairs, of other pairs whose images share a class set, and of the rest."""
    n, k, d = caption_emb.shape
    text = caption_emb.reshape(n * k, d)
    text_group = torch.as_tensor(np.repeat(groups, k))
    text_owner = torch.arange(n).repeat_interleave(k)
    g = torch.as_tensor(groups)
    sums = {"pos": 0.0, "same_class_neg": 0.0, "other_neg": 0.0}
    counts = dict.fromkeys(sums, 0)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = (image_emb[start:stop] @ text.T).double()
        own = text_owner[None, :] == torch.arange(start, stop)[:, None]
        same = (text_group[None, :] == g[start:stop, None]) & (g[start:stop, None] >= 0) & ~own
        other = ~own & ~same
        for key, mask in (("pos", own), ("same_class_neg", same), ("other_neg", other)):
            sums[key] += scores[mask].sum().item()
            counts[key] += int(mask.sum())
    return {key: sums[key] / counts[key] if counts[key] else float("nan") for key in sums}


def count_class_words(caption: str) -> int:
    return sum(word.lower() in COCO_CLASS_WORDS for word in _WORD.findall(caption))


def drop_words(caption: str, kind: str, k: int, rng: random.Random) -> str | None:
    """The caption with k random words of `kind` ("content" or "stop") deleted, or None with fewer than k."""
    words = caption.replace(".", " ").split()
    pick = [i for i, w in enumerate(words) if is_content_word(w) == (kind == "content") and w.isalpha()]
    if len(pick) < k:
        return None
    drop = set(rng.sample(pick, k))
    return " ".join(w for i, w in enumerate(words) if i not in drop)
```

- [ ] **Step 4: Run** `python -m pytest -q tests/test_diagnostics.py` → PASS (fix the implementation, not the expected values, unless a test's arithmetic is wrong; explain any such change).

- [ ] **Step 5: Commit** `feat(diagnostics): Stage 0 metric functions`.

---

### Task 10: Stage 0 diagnostics entry (`scripts/diagnose.py`, `scripts/run_diagnostics.sh`)

**Files:**
- Create: `scripts/diagnose.py`, `scripts/run_diagnostics.sh`, `tests/test_diagnose_script.py`

**Interfaces:**
- Consumes: Task 8's `evaluate_vwsd`; Task 9's functions; `mmae.engine.retrieval.encode_retrieval_set`, `retrieval_metrics`; `mmae.engine.eccv.build_extended_metrics`, `map_coco_ids`; `mmae.data.coco.retrieval_items`.
- Produces: `python scripts/diagnose.py [hydra overrides] +diag.runs_root=<dir> +diag.out=<dir> [+diag.probe_captions=5000] [+diag.zeroshot=true]` writing `<out>/diagnostics.json` and `<out>/embeddings/<run>.pt`; `bash scripts/run_diagnostics.sh <overrides>` (adds `data=coco_cluster`).

Behaviour:
1. Runs: every subdirectory of `diag.runs_root` holding `checkpoints/best.pt` and `config.yaml`, plus `zeroshot` (the pretrained backbone of `cfg.model`) when `diag.zeroshot` is true (default true).
2. Per run (model on `cuda` if available, else CPU): build `MultiMAE(run_cfg.model, run_cfg.data.max_text_len)`, load `best.pt`; encode the COCO test retrieval set (`CocoRetrieval` with `cfg.data` paths, `Collator(processor, max_text_len).retrieval`, `encode_retrieval_set(model, loader)`); save `{"image": fp16 (N,E), "caption": fp16 (N,K,E), "model": name, "seed": seed}`; record `logit_scale` (exp of the parameter, or null); VWSD metrics when `cfg.eval.vwsd_dir`; the masked-caption probe on the first caption of the first `diag.probe_captions` images: for kind in (content, stop) and k in (1, 2), the mean change in cosine to the own image and the t2i R@1 of the variant among all test images, over captions that have k words of that kind (`drop_words`, `random.Random(0)`).
3. When the PM ground truth is available (`cfg.data.pm_dir` set and `build_extended_metrics(cfg, "test", True)` not None): `class_set_groups`; per run `neighbour_purity` for images (k=10) and captions (k=10, owners = image row); `similarity_stats`; per-query PMRP t2i and i2t (`pmrp_rows`, `per_query_rprecision`), stored in `per_query.pt`, with the mean printed beside run.json's `test/pmrp` (or `eval.test.pmrp`) when present; per-run mean t2i PMRP by number of class words in the caption (0, 1, 2, 3+).
4. Tower swaps: for every run whose model name is not `contrastive`, the contrastive run with the same seed (from each run's `config.yaml`): retrieval metrics (and extended metrics when available) for (its image embeddings, the contrastive caption embeddings) and (contrastive image, its captions). Skip with a log line when no contrastive run has that seed.
5. Write `diagnostics.json`: `{"runs": {name: {...}}, "swaps": {"<run>__img+contrastive_txt": {...}, ...}}`, all floats.

- [ ] **Step 1: Failing smoke test** — create `tests/test_diagnose_script.py`:

```python
import json
import os
import subprocess
import sys

from helpers import REPO, make_fake_vwsd, run_train


def test_diagnose_on_two_tiny_runs(tmp_path, fake_coco):
    for model in ("contrastive", "fusion_concat"):
        result = run_train(tmp_path, fake_coco, f"model={model}", "train.save=best")
        assert result.returncode == 0, result.stderr[-3000:]
    runs_root = next((tmp_path / "res").rglob("checkpoints")).parents[1]
    images_dir, annotations_dir = fake_coco
    out = tmp_path / "diag"
    cmd = [sys.executable, str(REPO / "scripts" / "diagnose.py"),
           f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
           "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
           "eval.extended_metrics=false", f"eval.vwsd_dir={make_fake_vwsd(tmp_path / 'vwsd')}",
           f"+diag.runs_root={runs_root}", f"+diag.out={out}", "+diag.probe_captions=6"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1200,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"})
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads((out / "diagnostics.json").read_text())
    assert set(report["runs"]) >= {"zeroshot"} and len(report["runs"]) == 3
    for info in report["runs"].values():
        assert "vwsd/hit1" in info["vwsd"] and "probe" in info
    assert any("img+contrastive_txt" in key for key in report["swaps"])
    assert len(list((out / "embeddings").glob("*.pt"))) == 3
```

(If `run_train` with `train.save=best` does not write a checkpoint because `train/debug.yaml` sets `save: none`, the override above re-enables it; confirm in the run folder. Both tiny runs use `seed=42`, so the swap pairs them.)

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `scripts/diagnose.py`:

```python
"""Stage 0 diagnostics: encode every trained run under +diag.runs_root (and zero-shot CLIP) on the COCO 5k test
split and VWSD, then write tower swaps, class-set purity, similarity statistics, per-query PMRP and the
masked-caption probe to +diag.out/diagnostics.json (spec 2026-10-03, section 5).

  python scripts/diagnose.py data=coco_cluster +diag.runs_root=/local/wding/res/MultiMAE/coco/multimae/default \
      +diag.out=/local/wding/res/MultiMAE/coco/diagnostics/stage0 eval.vwsd_dir=/local/wding/Dataset/vwsd
"""
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from eccv_caption import Metrics  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mmae.data import CocoRetrieval, Collator, build_image_transform  # noqa: E402
from mmae.data.coco import retrieval_items  # noqa: E402
from mmae.engine.diagnostics import (  # noqa: E402
    class_set_groups, count_class_words, drop_words, neighbour_purity, per_query_rprecision, pmrp_rows,
    similarity_stats,
)
from mmae.engine.eccv import build_extended_metrics  # noqa: E402
from mmae.engine.retrieval import encode_retrieval_set, retrieval_metrics  # noqa: E402
from mmae.engine.vwsd import evaluate_vwsd  # noqa: E402
from mmae.models import MultiMAE  # noqa: E402
from mmae.models.backbones import processor_name  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("diagnose")


def find_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "checkpoints" / "best.pt").is_file() and (p / "config.yaml").is_file())


def load_model(cfg: DictConfig, run_dir: Path | None, device: torch.device):
    """(model, model config, max_text_len, seed, model type) of a run folder, or of zero-shot CLIP for None."""
    if run_dir is None:
        model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
        return model.to(device).eval(), cfg.model, cfg.data.max_text_len, None, "zeroshot"
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    model = MultiMAE(run_cfg.model, max_text_len=run_cfg.data.max_text_len)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")["model"])
    return model.to(device).eval(), run_cfg.model, run_cfg.data.max_text_len, int(run_cfg.seed), str(run_cfg.model.name)


@torch.no_grad()
def embed_texts(model, collator, texts: list[str], device, batch: int = 256) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), batch):
        tokens = collator.tokenize(texts[start : start + batch])
        chunks.append(model.embed_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device)).float().cpu())
    return torch.cat(chunks)


@torch.no_grad()
def probe(model, collator, items, image_emb: torch.Tensor, n_images: int, device) -> dict[str, float]:
    """Delete k content (or stop) words from each image's first caption: mean change in cosine to the own image,
    and t2i R@1 of the shortened caption among all test images."""
    rng = random.Random(0)
    out: dict[str, float] = {}
    for kind in ("content", "stop"):
        for k in (1, 2):
            rows, short, full = [], [], []
            for row, (_, captions) in enumerate(items[:n_images]):
                text = drop_words(captions[0], kind, k, rng)
                if text is not None:
                    rows.append(row), short.append(text), full.append(captions[0])
            if not rows:
                continue
            idx = torch.tensor(rows)
            short_emb, full_emb = embed_texts(model, collator, short, device), embed_texts(model, collator, full, device)
            own = image_emb[idx]
            out[f"{kind}{k}/delta_cos"] = ((short_emb * own).sum(-1) - (full_emb * own).sum(-1)).mean().item()
            out[f"{kind}{k}/t2i_R1"] = 100.0 * ((short_emb @ image_emb.T).argmax(dim=1) == idx).double().mean().item()
            out[f"{kind}{k}/n"] = float(len(rows))
    return out


def encode_run(cfg, diag, name, run_dir, items, out: Path, device) -> tuple[dict, tuple]:
    model, model_cfg, max_len, seed, kind = load_model(cfg, run_dir, device)
    processor = processor_name(model_cfg.backbone)
    transform, collator = build_image_transform(processor), Collator(processor, max_len)
    dataset = CocoRetrieval(cfg.data.images_dir, cfg.data.annotations_dir, "test", transform, cfg.data.limit_test)
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, num_workers=cfg.train.num_workers,
                        collate_fn=collator.retrieval)
    batches = ({key: value.to(device) for key, value in batch.items()} for batch in loader)
    image_emb, caption_emb = (t.cpu() for t in encode_retrieval_set(model, batches))
    info = {
        "model": kind, "seed": seed,
        "logit_scale": float(model.logit_scale.exp()) if model.logit_scale is not None else None,
        "test": retrieval_metrics(image_emb, caption_emb),
        "probe": probe(model, collator, items, image_emb, int(diag.get("probe_captions", 5000)), device),
    }
    if cfg.eval.get("vwsd_dir"):
        info["vwsd"] = evaluate_vwsd(model, cfg.eval.vwsd_dir, transform, collator, lang=cfg.eval.vwsd_lang,
                                     prompt=cfg.eval.vwsd_prompt, batch_size=cfg.train.eval_batch_size,
                                     num_workers=cfg.train.num_workers, device=device)
    torch.save({"image": image_emb.half(), "caption": caption_emb.half(), "model": kind, "seed": seed},
               out / "embeddings" / f"{name}.pt")
    log.info("%s (%s, seed %s): rsum %.2f", name, kind, seed, info["test"]["rsum"])
    return info, (image_emb, caption_emb, kind, seed)


def class_set_report(report, embeddings, extended, items, out: Path) -> None:
    """Purity, similarity statistics and per-query PMRP (needs the PM ground truth)."""
    pm_gts = Metrics(extra_file_dir=str(extended.pm_dir)).pm_gts
    ids_img, ids_cap = extended.image_ids, extended.caption_ids
    groups = class_set_groups(pm_gts, ids_img, ids_cap)
    rows = pmrp_rows(pm_gts, ids_img, ids_cap)
    k = ids_cap.shape[1]
    owners = np.repeat(np.arange(len(ids_img)), k)
    class_words = np.array([count_class_words(c) for _, captions in items for c in captions])
    per_query = {}
    for name, (img, cap, _, _) in embeddings.items():
        text = cap.reshape(-1, cap.shape[-1])
        t2i = per_query_rprecision(text[rows["t2i"][0]], img, rows["t2i"][1])
        i2t = per_query_rprecision(img[rows["i2t"][0]], text, rows["i2t"][1])
        per_query[name] = {"t2i": torch.from_numpy(t2i), "i2t": torch.from_numpy(i2t)}
        words = class_words[rows["t2i"][0]]
        by_words = {}
        for label, select in (("0", words == 0), ("1", words == 1), ("2", words == 2), ("3+", words >= 3)):
            if select.any():
                by_words[label] = 100.0 * float(t2i[select].mean())
        report["runs"][name].update(
            purity_i2i=neighbour_purity(img, groups),
            purity_t2t=neighbour_purity(text, np.repeat(groups, k), owners=owners),
            similarity=similarity_stats(img, cap, groups),
            pmrp_per_query={"t2i": 100.0 * t2i.mean(), "i2t": 100.0 * i2t.mean(), "mean": 50.0 * (t2i.mean() + i2t.mean())},
            pmrp_t2i_by_class_words=by_words,
        )
    torch.save(per_query, out / "per_query.pt")


def tower_swaps(report, embeddings, extended) -> None:
    contrastive = {seed: name for name, (_, _, kind, seed) in embeddings.items() if kind == "contrastive"}
    for name, (img, cap, kind, seed) in embeddings.items():
        if kind in ("contrastive", "zeroshot"):
            continue
        partner = contrastive.get(seed)
        if partner is None:
            log.info("no contrastive run with seed %s: no tower swap for %s", seed, name)
            continue
        partner_img, partner_cap = embeddings[partner][:2]
        for key, (i, c) in {f"{name}__img+contrastive_txt": (img, partner_cap),
                            f"contrastive_img+{name}__txt": (partner_img, cap)}.items():
            metrics = retrieval_metrics(i, c)
            if extended is not None:
                metrics.update(extended(i, c))
            report["swaps"][key] = metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    diag = cfg.get("diag") or {}
    out = Path(diag["out"])
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = retrieval_items(cfg.data.annotations_dir, "test")[: cfg.data.limit_test]
    runs = [(p.name, p) for p in find_runs(Path(diag["runs_root"]))]
    if diag.get("zeroshot", True):
        runs.insert(0, ("zeroshot", None))
    extended = build_extended_metrics(cfg, "test", two_modalities=True)
    report: dict = {"runs": {}, "swaps": {}}
    embeddings = {}
    for name, run_dir in runs:
        report["runs"][name], embeddings[name] = encode_run(cfg, diag, name, run_dir, items, out, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if extended is not None and extended.pm_dir is not None:
        class_set_report(report, embeddings, extended, items, out)
    else:
        log.info("no PM ground truth: purity, similarity statistics and per-query PMRP skipped")
    tower_swaps(report, embeddings, extended)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("wrote %s", out / "diagnostics.json")


if __name__ == "__main__":
    main()
```

(`encode_retrieval_set` iterates any iterable of batches, so the generator that moves batches to the device works
without an Accelerator. `retrieval_items` returns `(image path, captions)` pairs, the order `CocoRetrieval` uses.)

Create `scripts/run_diagnostics.sh` (and `chmod +x`):

```bash
#!/usr/bin/env bash
# Stage 0 diagnostics on a cluster node (COCO on the node's local disk), single process.
# Usage: scripts/run_diagnostics.sh +diag.runs_root=<dir> +diag.out=<dir> [eval.vwsd_dir=<dir>] [overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/diagnose.py data=coco_cluster "$@"
```

- [ ] **Step 4: Run** `python -m pytest -q tests/test_diagnose_script.py` → PASS.

- [ ] **Step 5: Full suite, commit** `feat(diagnostics): Stage 0 entry script and cluster wrapper`.

---

### Task 11: Documentation

**Files:**
- Modify: `README.md` (configs/commands sections), `CLAUDE.md` (gitignored, local: Commands, Architecture, Configs, Testing)

- [ ] **Step 1:** Document every new switch with its default and its arm id (M1 `model.mlm_image_source`, M2b `model.masking.text_mode`, M3 `model.pooled_conditioning`, M6 `model.loss.weights.masked_view`, R2 `train.lr_text/lr_vision/layer_decay/freeze_vision_epochs`, `train.seeded_sampler`), VWSD (`eval.vwsd_dir`, data at `/data/SSD/vwsd`, node `/local/wding/Dataset/vwsd`, CC-BY-NC 4.0), the diagnostics command (`scripts/run_diagnostics.sh`), `tests/test_model_variants.py` (VARIANTS: add a switch there), and the run registry `tests/20261003_ml_improve/runs.md`. Keep the existing style: short, factual.
- [ ] **Step 2:** `python scripts/check_reports_sum.py` → `reports_sum.md: OK`; commit README only (`docs: document the multilearner-line switches`).

---

## Self-review notes

- Spec coverage: section 5 (E0a VWSD: Tasks 8, 10; E0b tower swap: Task 10; E0c purity, E0d PMRP by query type, E0e similarity statistics: Tasks 9, 10; E0f masked-caption probe: Tasks 9, 10); section 6 code arms M1 (Task 2), M2b (Task 3), M3 (Task 4), M6 (Task 5), R2 (Task 6); section 7 seeded sampler (Task 7); section 8 testing rules (Global Constraints, guard checks in Tasks 2 to 7).
- M5 and M2a are config-only (spec section 6) and need no task.
- Operations outside the code (controller, not implementers): add `/local/wding/Dataset/vwsd:/data/SSD/vwsd` to the cluster `DATA_MAP`, sync VWSD with `cluster check --sync-data` on a `train.py ... eval.vwsd_dir=/local/wding/Dataset/vwsd` command, commit with `cluster run` in the subject before each launch.
