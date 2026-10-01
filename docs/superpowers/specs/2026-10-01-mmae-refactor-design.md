# MultiMAE v1 refactor: design

Date: 2026-10-01. Branch: `refactor`. The pre-refactor code (v0) is frozen at tag `legacy-v0` and branch `legacy` (commit `1b6603e`).

## 1. Goal

Rewrite the codebase so that the fusion masked autoencoder is **correct, simple and extendable**. The project has just started, so reproducing v0's numbers is not a goal; several of v0's numbers are known to be wrong (no input masking, frozen encoders, random projection heads). The fusion design space (fusion types, decoders, loss mixes) will be explored after this refactor, so the code must make new variants cheap to add.

Kept: Hydra configs and wandb logging. Dropped: timm, PyTorch Lightning (never used here), the CIFAR and wikitext pipelines.

## 2. Decisions

| Topic | Decision |
|---|---|
| Main line | Fusion MAE on COCO. Single-modality MAE/MLM and the parallel (no-fusion) model stay as config variants for debugging and optional comparison, not as fixed baselines. |
| Encoders | HF `transformers` CLIP, both towers fine-tuned by default (`model.freeze_backbones: false`). A small registry keeps the door open for non-CLIP backbones. |
| Image masking | MAE-style token dropping before the vision transformer, ratio 0.75. |
| Text masking | Learned `[MASK]` embedding on the input, ratio 0.15, never on BOS/EOS/padding. |
| Fusion | `none`, `concat` (optional transformer depth) and `multilearner`, behind one interface. |
| Losses | Contrastive is the main loss, on a clean (unmasked) pass, with CLIP's pooling, pretrained projections and learnable logit scale. MAE and MLM on the masked pass, masked positions only. |
| Benchmark | COCO 5k retrieval (i2t, t2i). |
| Training | Accelerate (single node, multi-GPU), plain hand-written loop. |
| Early stopping | `val/retrieval/rsum` (max) for models with both modalities; `val/loss` (min) for single-modality models. Set per model config (`model.monitor`). |
| Learning rates | 1e-5 for the towers, 1e-4 for new modules. **Untested placeholders**; tuning is out of scope. |
| Outputs | One folder per run under `res/`, not Hydra's `outputs/`. |
| Package | `src/` is renamed to `mmae/`; `setup.py` packages `mmae`. |

## 3. Repository layout

```
MultiMAE/
├── train.py                  # Hydra entry: accelerate launch train.py model=fusion_concat ...
├── evaluate.py               # retrieval eval of a run's best checkpoint, or of zero-shot CLIP
├── setup.py                  # packages mmae; install_requires from requirements.txt; extras_require[dev] = pytest
├── requirements.txt          # pinned to the MultiMAE env versions
├── configs/
│   ├── config.yaml           # defaults list, seed, wandb, paths
│   ├── model/                # fusion_concat, fusion_multilearner, fusion_none, image_mae, text_mlm
│   ├── data/                 # coco, coco_cluster
│   └── train/                # default, debug
├── mmae/
│   ├── data/                 # coco.py, transforms.py, collate.py
│   ├── models/               # backbones.py, masking.py, fusion.py, decoders.py, model.py
│   ├── losses.py
│   ├── engine/               # trainer.py, retrieval.py
│   └── utils/                # run.py (res/ run folder), logging.py, seed.py, dist.py
├── scripts/                  # run_local.sh, run_cluster.sh, list_runs.py, check_reports_sum.py
├── tests/                    # pytest; dated debug folders (yyyymmdd_<name>/) also live here
└── docs/                     # see section 11
```

Removed from the tree (kept in `legacy-v0`): `src/`, `main_*.py`, the v0 configs, `scripts/check_accelerate.sh`, `scripts/run_fusion_mmae*.sh`, `notebook/`, `tests/.gitkeep`, `src/test/20261001_*` (their logs are copied into `docs/reports/auto/v0/pilots/`).

## 4. Data (`mmae/data/`)

- **`CocoPairs(split)`**: one (image, caption) pair per item. `train` reads `coco_karpathy_train.json`. `val` and `test` flatten the 5-caption files `coco_karpathy_{val,test}.json` on the fly, so the derived `*_one_caption.json` files and the conversion notebook are no longer needed.
- **`CocoRetrieval(split)`**: one item per image with its 5 captions, from `coco_karpathy_{val,test}.json`.
- **Paths** are explicit config keys `data.images_dir` and `data.annotations_dir`. `data/coco_cluster.yaml` overrides them for the cluster.
- **Images**: PIL, `ImageFile.LOAD_TRUNCATED_IMAGES = True`. Preprocessing is OpenAI CLIP's own torchvision pipeline (resize shortest side to 224 bicubic, center crop 224, CLIP mean/std), with the sizes and statistics read from the backbone's image processor config. HF's `CLIPImageProcessor` resizes slightly differently (measured 2026-10-01: mean absolute difference 0.03, up to 2.9 on single pixels); the published zero-shot numbers use OpenAI's pipeline, so we do too. No augmentation.
- **Text**: tokenized per batch in the collate function with the backbone's tokenizer: `max_length = data.max_text_len` (32), `truncation=True`, `padding="max_length"`, returning `attention_mask` and `special_tokens_mask`. CLIP's tokenizer keeps EOS when truncating; this is tested.
- **Debug subsets**: `data.limit_train`, `data.limit_val`, `data.limit_test` (null = all) take the first N items (deterministic). They replace the monkeypatching the v0 harnesses needed.
- No global side effects at import (v0 set `HF_DATASETS_CACHE`).

## 5. Model (`mmae/models/`)

### 5.1 Towers (`backbones.py`)

Each tower exposes:
- `encode(inputs, mask=None) -> tokens`: hidden states for the decoders. Image: `(B, 1 + N_visible, C_v)`. Text: `(B, T, C_t)`.
- `embed(inputs) -> (B, D)`: L2-normalized joint-space embedding of a clean input, used by the contrastive loss and retrieval.

`build_backbone(cfg.model.backbone)` returns `(vision_tower, text_tower, logit_scale)` from a registry keyed by `backbone.type` (`hf_clip` for now). Both towers load from one checkpoint, `backbone.pretrained` (default `openai/clip-vit-base-patch32`), because the pretrained projections only share a space within one model. The checkpoint's patch size, grid and dims come from its config; nothing is hard-coded (v0 hard-coded 16 px patches on a 32 px backbone).

**Vision, `hf_clip`.** Compute CLIP's embeddings (patch, CLS, position). With a mask, keep CLS plus the visible patch tokens (gathered by `ids_keep`), then run `pre_layrnorm` and the encoder on that shorter sequence. `encode` returns the encoder's last hidden state. `embed` = CLS → `post_layernorm` → `visual_projection` → normalize.

**Text, `hf_clip`.** Token embeddings, then masked positions replaced by a learned `mask_embedding` (shape `C_t`, trainable even when towers are frozen), then position embeddings, CLIP's causal mask combined with the tokenizer's `attention_mask`, the encoder and `final_layer_norm`. `encode` returns the final-layer-normed hidden states. `embed` = hidden state at the EOS position (located as HF does) → `text_projection` → normalize.

**Logit scale.** CLIP's pretrained `logit_scale` parameter, learnable, exponentiated and clamped at 100 as in CLIP.

**Required equivalences (tested).** `embed` equals HF's `CLIPModel.get_image_features` / `get_text_features` (normalized). `encode` with an all-visible mask (or no masked tokens) equals `encode` without a mask, and equals HF's own `last_hidden_state` for the same inputs. These pin down the code that calls HF's internal stages directly.

**Freezing.** `model.freeze_backbones: true` sets `requires_grad=False` on every tower parameter except `mask_embedding`. Modules after the towers (fusion projections, decoders, the `mean`-pooling projection, `logit_scale`) stay trainable.

**Pooling option.** `model.pooling: native | mean`. `native` (default) is described above. `mean` mean-pools the tower's clean `encode` output over non-padded tokens and applies a freshly initialized linear projection to the joint dimension, for comparison with v0's mean-pooling experiments.

### 5.2 Masking (`masking.py`)

Pure functions, no module state:
- `random_patch_mask(batch, num_patches, ratio, generator=None) -> (ids_keep, mask)`: per-sample random (argsort of uniform noise, as in MAE). `num_masked = int(num_patches * ratio)`; `mask` is `(B, N)` bool, True = masked; `ids_keep` lists visible patch indices in ascending order.
- `random_token_mask(attention_mask, special_tokens_mask, ratio, generator=None) -> mask`: per sample, among positions with `attention_mask == 1` and `special_tokens_mask == 0`, mask `max(1, round(ratio * n_maskable))` positions. Captions with no maskable token get no mask.

### 5.3 Fusion (`fusion.py`)

Interface: `fusion(image_tokens, text_tokens, text_padding) -> FusionOutput(image_memory, image_memory_padding, text_memory, text_memory_padding)`. Either modality may be `None` for single-modality models. Padding masks are True at padded positions and are passed to every attention layer that reads the memory (v0 let decoders attend to pads).

Before fusion, image tokens and text tokens are each projected to `fusion.dim` (256) by a linear layer.

- **`none`**: image memory = projected image tokens; text memory = projected text tokens.
- **`concat`**: add a learned modality-type embedding (2 × `dim`), concatenate along the sequence, then `fusion.depth` pre-norm transformer encoder layers (`fusion.heads`, MLP ratio 4). `depth: 0` (default) is plain concatenation, the v0 `MultiModalFusionMAE_CLIP` design. Both decoders read the fused sequence.
- **`multilearner`**: concat as above, then three learner transformers (image, text, joint; each `learner.depth` = 2 layers, `learner.ff_dim` = 512). Image memory = MLP([image ; joint]), text memory = MLP([text ; joint]), each MLP `2·dim → 512 → dim`. This is v0's `MultiModalFusionMAE_CLIP_MultiLearner` structure.

New fusion types are added by writing one class with this interface and registering it under a name.

### 5.4 Decoders (`decoders.py`)

Both are query decoders: learned queries plus learned position embeddings, a `nn.TransformerDecoder` (`decoder.depth` 4, `decoder.heads` 8, `decoder.dropout` 0.1, batch-first) cross-attending to the memory with its padding mask.
- **Image**: one query per patch position (`N` = grid size squared of the backbone), output `p·p·3` pixels per patch.
- **Text**: one query per token position (`max_text_len`), with the caption's padding as `tgt_key_padding_mask`, output vocabulary logits.

### 5.5 Model (`model.py`)

`MultiMAE(cfg)` assembles towers, fusion and decoders according to `model.modalities` (`[image, text]`, `[image]` or `[text]`). `forward(batch) -> dict`:
1. **Clean pass** (only when both modalities are present): `img = vision.embed(images)`, `txt = text.embed(text)`, `loss_contrastive`.
2. **Masked pass**: sample masks, `encode` the visible inputs, fuse, decode; `loss_mae` on masked patches, `loss_mlm` on masked tokens.
3. `loss = Σ weight_k · loss_k` over the losses present (`loss.weights.{contrastive, mae, mlm}`, default 1.0 each).

The dict holds `loss` and every individual loss. Because all losses are computed inside `forward`, DDP needs no unwrapping. `model.embed_image` / `model.embed_text` are exposed for retrieval.

## 6. Losses (`mmae/losses.py`)

- **Contrastive**: symmetric InfoNCE, `logits = scale · img @ txt.T`. With more than one process and `loss.gather: true`, embeddings are all-gathered with gradients (`torch.distributed.nn.functional.all_gather`), and each rank computes the loss of its local rows against the global batch (open_clip's local-loss formulation). With one process, plain InfoNCE.
- **MAE**: target = patchified, preprocessed input image. With `loss.norm_pix: true` (default, as in the MAE paper) each target patch is normalized by its own mean and variance (eps 1e-6). Loss = mean over masked patches of the per-patch mean squared error.
- **MLM**: cross-entropy between text logits and the original token ids at masked positions only.
- v0's `PhaseTraining` schedule is dropped (all its weights were 1.0).

## 7. Training (`mmae/engine/trainer.py`, `train.py`)

- **Accelerate**: `Accelerator(mixed_precision=train.precision, gradient_accumulation_steps=train.grad_accum)`. `train.precision`: `no` (default) or `bf16`. Launch: `accelerate launch --num_processes N train.py ...`, single node; `python train.py` for one GPU.
- **Optimizer**: AdamW, two groups: tower parameters at `train.lr_backbone` (1e-5), everything else at `train.lr` (1e-4). No weight decay on biases, norm weights, embeddings (position, type, mask, queries) and `logit_scale`; `train.weight_decay` (0.05) elsewhere.
- **Schedule**: per optimizer step, linear warmup over `train.warmup_steps` (default 500) then cosine decay to 0.
- **Gradient clipping**: `train.grad_clip` (default 1.0; null disables).
- **Loop**: for each epoch: train; every `train.eval_every` epochs (1) compute val losses on `CocoPairs(val)` and, for two-modality models, val retrieval on `CocoRetrieval(val)`; update early stopping on `model.monitor` (metric and mode, set per model config because it depends on the modalities) with `train.patience` (5) and `train.min_delta`; keep a CPU copy of the best weights. After the last epoch or an early stop: restore the best weights, then compute test losses and test retrieval.
- **Logging**: every `train.log_every` (50) steps, the averaged train losses and learning rates. Per eval, val metrics. At the end, test metrics. Metric names: `train/*`, `val/*`, `val/retrieval/*`, `test/*`, `test/retrieval/*`, with `epoch` logged as a metric and the global step as wandb's step.
- **Eval under multi-GPU**: eval loaders are prepared (sharded); retrieval gathers embeddings per batch with `gather_for_metrics`, captions shaped `(B, 5, D)`, so padded duplicates are dropped; every rank gets identical metrics and makes the same stopping decision. Val losses are averaged across processes.
- **Seeding**: `accelerate.utils.set_seed(cfg.seed)`; data workers seeded from it.
- **Checkpoints**: `train.save: best | none` (default `best`). `best.pt` holds the unwrapped model's full `state_dict`, the resolved config, the epoch and the val metrics. Resuming is out of scope.

## 8. Run folder (`mmae/utils/run.py`)

Adapted from CoSiR's `ExperimentManager`, keeping its path layout, lifecycle and summary, and fixing its problems (a tag `KeyError` after the registry reloads, read-modify-write races on the shared registry, `load_experiment` overwriting the saved config, eleven folders created up front, one JSON file per epoch, `plt.show()`, no multi-process awareness, no git or wandb record).

```
res/<wandb.project>/<wandb.group or "default">/<YYYYMMDD_HHMMSS>_<run name>/   # suffix _1, _2 if taken
├── config.yaml      # resolved Hydra config
├── run.json         # status (running|completed|failed), created/started/ended/duration, git commit + dirty flag,
│                    # command line, host, num processes, wandb id/url, best epoch, final val/test metrics
├── metrics.jsonl    # one line per log call: {"step", "epoch", "metrics": {...}}
├── train.log        # console log of the main process
├── error.txt        # traceback, only when the run failed
├── checkpoints/     # best.pt; created on first write
└── plots/           # learning curves drawn from metrics.jsonl at the end (matplotlib, Agg backend)
```

- `paths.res_dir` (default `res`) is the root; `res/` is gitignored.
- `Run` is a context manager used by `train.py`. Only the main process creates the folder and writes files; other ranks get a no-op object.
- The run name is `wandb.name` if set, else the model config name.
- There is no shared registry file. `scripts/list_runs.py` scans `run.json` files and lists or filters runs by group, status, tag or model.
- Hydra writes nothing to disk (`hydra.run.dir: .`, no output subdir, Hydra's file logging disabled); `train.log` replaces it.

## 9. Evaluation (`mmae/engine/retrieval.py`, `evaluate.py`)

- Embeddings: `model.embed_image` for each image, `model.embed_text` for each of its 5 captions.
- Ranks are computed by counting scores strictly greater than the positive's score (i2t: best rank over the 5 positive captions; t2i: rank of the one positive image), not by a full argsort.
- Metrics, same as v0: `i2t_R1/R5/R10`, `t2i_R1/R5/R10`, `i2t_meanR/medR`, `t2i_meanR/medR`, `i2t_mAP`, `t2i_mAP`. **Changed:** `rsum` = R@1 + R@5 + R@10 in both directions (six terms, the standard definition); v0's `r_sum` summed R@1 and R@5 only.
- `evaluate.py run_dir=<path>` loads that run's `config.yaml` and `checkpoints/best.pt`; `evaluate.py` without a run dir evaluates the pretrained backbone (zero-shot CLIP). Results print and, with a run dir, are written to `run.json`.
- **Known-answer check:** zero-shot OpenAI CLIP ViT-B/32 on the COCO 5k test split should give about i2t R@1 50.1 and t2i R@1 30.4 (published numbers). A large deviation means preprocessing, tokenization, pooling or the metrics are wrong. (v0's `CLIP_Retrieval_Metrics`, 58.4 / 37.8, are CLIP ViT-L/14@336 numbers.)

## 10. Testing (`tests/`, pytest)

Fast unit tests use a tiny randomly initialized CLIP built from a small `CLIPConfig`, so they need no download or GPU. Tests that load the real checkpoint, COCO data or a GPU are marked `slow`.

| Area | Checks |
|---|---|
| Towers | `embed` equals HF `get_*_features`; masked `encode` with nothing masked equals clean `encode` and HF `last_hidden_state`; tiny and real (slow) checkpoint. |
| Masking | Exact counts; never BOS, EOS or padding; at least one token per caption with maskable tokens; masks differ across samples; reproducible from a generator seed. |
| Padding | Changing the contents of padded positions does not change any decoder output. |
| Data | Truncation keeps EOS; val/test flattening yields 5 pairs per image; `limit_*` works; grayscale, CMYK and truncated images load as RGB; preprocessing maps a constant colour to CLIP-normalized values (end-to-end correctness comes from the zero-shot check in section 9). |
| Losses | Contrastive equals a direct reference implementation; the gathered version under 2 CPU processes (gloo) equals single-process InfoNCE on the concatenated batch; MAE/MLM only count masked positions. |
| Retrieval | Metrics equal v0's `evalrank` (copied into the test as a reference) on the same random and real embeddings; 1 vs 2 processes identical with a set size not divisible by world size × batch. |
| Model configs | For each of `fusion_concat`, `fusion_multilearner`, `fusion_none`, `image_mae`, `text_mlm`: finite losses; gradients reach exactly the expected parameters (none in the towers when frozen); a tiny-batch overfit run lowers MAE and MLM losses. |
| Smoke (slow) | `train.py train=debug` per model config on GPU; one 2-process CPU DDP run (`ACCELERATE_USE_CPU=1`) of `fusion_concat`; the run folder contains the expected files. |

For the key guards (masking counts, padding, gather, retrieval maps, HF equivalence), we confirm the test fails when the guarded code is broken on purpose.

## 11. Docs and reports

Follows CoSiR's structure and the reports layout rule:

```
docs/
├── superpowers/specs/      # this spec
├── superpowers/plans/      # the implementation plan
├── reports/
│   ├── reports_sum.md      # index; one row per report; checked by scripts/check_reports_sum.py
│   ├── auto/v0/pilots/20261001_fusion_ddp_fixes/      # copied logs (docs only) from legacy-v0
│   ├── auto/v0/pilots/20261001_legacy_entrypoints/
│   ├── auto/v1/            # reports on the refactored code
│   └── stage/ weekly/ pptx/ assets/                   # created when first needed
└── archive/templates/      # v0's docs/templates (CHANGELOG/TODO templates)
```

Research lines: **v0** (code up to `legacy-v0`) and **v1** (this refactor). `scripts/check_reports_sum.py` is ported from CoSiR and adapted to these paths. `README.md` and the local `CLAUDE.md` are rewritten for the new layout. Code comments and docstrings are in English. `TODO.md` stays as the dev log.

## 12. Configs (sketch)

```yaml
# configs/config.yaml
defaults: [model: fusion_concat, data: coco, train: default, _self_]
seed: 42
paths: {res_dir: res}
wandb: {enabled: true, project: multimae, entity: augustoxq, group: "", name: "", tags: [], notes: "", mode: online}

# configs/model/fusion_concat.yaml
name: fusion_concat
modalities: [image, text]
backbone: {type: hf_clip, pretrained: openai/clip-vit-base-patch32}
freeze_backbones: false
pooling: native
masking: {image_ratio: 0.75, text_ratio: 0.15}
fusion: {type: concat, dim: 256, depth: 0, heads: 8}
decoder: {depth: 4, heads: 8, dropout: 0.1}
loss: {weights: {contrastive: 1.0, mae: 1.0, mlm: 1.0}, gather: true, norm_pix: true}
monitor: {metric: val/retrieval/rsum, mode: max}

# configs/train/default.yaml
epochs: 10
batch_size: 128
eval_batch_size: 256
lr: 1.0e-4          # untested placeholder
lr_backbone: 1.0e-5 # untested placeholder
weight_decay: 0.05
warmup_steps: 500
grad_clip: 1.0
grad_accum: 1
precision: "no"
eval_every: 1
patience: 5
min_delta: 1.0e-4
log_every: 50
num_workers: 8
save: best
```

`image_mae` and `text_mlm` set `modalities` to one entry, `fusion.type: none`, drop the absent losses and monitor `val/loss` (min). `data/coco.yaml` holds `images_dir: /data/SSD/coco/images`, `annotations_dir: /data/SSD/coco/annotations`, `max_text_len: 32` and the `limit_*` keys; `train/debug.yaml` sets 1 epoch, small limits and batch sizes, and `wandb.enabled: false`. The wandb project is new (`multimae`) so v1 runs don't mix with v0's.

## 13. Out of scope

New fusion or decoder designs, data augmentation, loss-weight schedules, hyperparameter tuning (including the two learning rates), resuming from checkpoints, non-CLIP backbones beyond the registry hook, multi-node training, and integration with the cluster CLI. `scripts/run_cluster.sh` keeps v0's arguments (batch size, epochs, note) and uses `data=coco_cluster`.

## 14. Migration

1. Done: v0 fixes committed on `main` (`1b6603e`), tagged `legacy-v0`, branch `legacy`, pushed.
2. Implement on `refactor` following the plan; subagents implement task by task with a review per task, then a final whole-branch review, one fix wave and a scoped re-review.
3. Merge `refactor` into `main` after the user approves.
