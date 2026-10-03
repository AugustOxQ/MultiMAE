# MultiMAE

MultiMAE is a research codebase for a fusion masked autoencoder built on CLIP towers. Both CLIP encoders (image and text) are fine-tuned and trained with three losses: a contrastive loss on clean inputs, an image MAE loss and a text MLM loss on masked inputs, where the masked image and text tokens are fused before two query decoders reconstruct them. The benchmark is COCO 5k image-text retrieval (Karpathy split). The code lives in the `mmae` package; the design is in [`docs/superpowers/specs/2026-10-01-mmae-refactor-design.md`](docs/superpowers/specs/2026-10-01-mmae-refactor-design.md).

## Install

```bash
conda activate MultiMAE        # /root/miniconda3/envs/MultiMAE
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e ".[dev]"        # the mmae package plus pytest
```

## Data

COCO images and the Karpathy split files are expected at

- `data.images_dir` (default `/data/SSD/coco/images`)
- `data.annotations_dir` (default `/data/SSD/coco/annotations`), holding `coco_karpathy_train.json`, `coco_karpathy_val.json` and `coco_karpathy_test.json`

Override the keys on the command line (`data.images_dir=...`). `data=coco_cluster` points both at the cluster nodes' local disk and also sets `paths.res_dir` to the node's results root (`/local/wding/res/MultiMAE/coco`).

The val and test files carry five captions per image. Retrieval uses each image with its five captions. For the validation and test losses they are flattened on the fly into caption-major pairs: every image with its first caption, then every image with its second caption, and so on. An unshuffled eval batch of up to 5000 pairs (the number of images) therefore never shows the same image twice, so no caption in it is a false negative for the contrastive loss. `data.limit_val` and `data.limit_test` keep the first N images (N <= 5000) for both: retrieval uses them with five captions, the loss pairs with their first caption.

## Training

```bash
python train.py model=fusion_concat                  # one GPU
python train.py train=debug                          # tiny smoke run, wandb off, no checkpoint
python train.py model=fusion_multilearner train.lr=2e-4 train.epochs=20
```

Model configs (`configs/model/`), all built on `base.yaml`:

| `model=` | What it trains |
|---|---|
| `fusion_concat` | Both modalities; the image and text tokens are concatenated (optionally through `fusion.depth` transformer layers) and both decoders read the fused sequence. Default. |
| `fusion_multilearner` | Both modalities; concatenation followed by an image learner, a text learner and a joint learner, each decoder reading its own MLP of own plus joint tokens. |
| `fusion_none` | Both modalities, no fusion: parallel contrastive, MAE and MLM, each decoder reads only its own modality. |
| `contrastive` | Baseline: the same CLIP fine-tune with the contrastive loss alone (`reconstruction: false`: no masked pass, fusion, projections or decoders). |
| `image_mae` | Image only: MAE on the CLIP vision tower (debugging, comparison). Monitors `val/loss`. |
| `text_mlm` | Text only: MLM on the CLIP text tower (debugging, comparison). Monitors `val/loss`. |

### Switches for the multilearner line

Every switch below defaults to today's behaviour (losses, masks, parameter groups and data order unchanged); old run configs without the keys still build. Arm ids are from `docs/superpowers/specs/2026-10-03-improve-multilearner-design.md`.

| Arm | Key | Default | Meaning |
|---|---|---|---|
| M1 | `model.mlm_image_source` | `masked` | Image the MLM decoder reads: `masked` (the 25% of patches the masked pass keeps), `clean` (all patches of the clean pass) or `clean_detached` (the same, no gradient into the vision tower through that path). |
| M2b | `model.masking.text_mode` | `random` | `content`: the masked text tokens are drawn only from content words (stop words, punctuation and numbers never; word list in `mmae/data/stopwords.py`), same count as `random`. A caption with no content word masks nothing. |
| M3 | `model.pooled_conditioning` | `false` | `true`: the text decoder also reads the clean image embedding and the image decoder the clean text embedding, one extra memory token each (never their own modality). |
| M6 | `model.loss.weights.masked_view` | `0.0` | Weight of an InfoNCE between the masked caption's pooled embedding and the clean images. |
| R2 | `train.lr_text`, `train.lr_vision` | `null` | Learning rate of the CLIP text and vision tower; `null` falls back to `train.lr_backbone`. |
| R2 | `train.layer_decay` | `1.0` | BEiT-style layer-wise lr decay inside each tower; `1.0` is none. |
| R2 | `train.freeze_vision_epochs` | `0` | The vision tower's lr is 0 for the first N epochs (counted in optimizer steps, so it holds with `grad_accum > 1`). |
| | `train.seeded_sampler` | `false` | `true`: the training shuffle uses a generator seeded by `seed` alone, so seed k gives every model type the same data order. |

Every new `train` key is in both `configs/train/default.yaml` and `configs/train/debug.yaml`. Run registry for this line: `tests/20261003_ml_improve/runs.md`.

`model.backbone.pretrained` names the CLIP checkpoint (default `openai/clip-vit-base-patch32`), and the tokenizer and image preprocessing follow it: `model.backbone.processor` defaults to `${.pretrained}`, except that `tiny-random-clip` (a tiny random CLIP for tests and CPU smoke runs) uses B/32's.

The contrastive loss uses CLIP's pretrained, learnable logit scale. After each optimizer step the trainer clamps the parameter to [0, ln 100], as open_clip does, so the scale stays at most 100 without cutting off its gradient; `train/logit_scale` logs the scale.

Launch scripts (both use `accelerate launch` on every visible GPU, or one process without a GPU):

```bash
scripts/run_local.sh "first try" model=fusion_multilearner train.epochs=20   # note, then Hydra overrides
scripts/run_cluster.sh 128 10 "baseline" model=fusion_concat                 # batch size, epochs, note, overrides
accelerate launch --num_processes 4 --multi_gpu train.py model=fusion_concat # by hand, one node
```

The note can be any text, quotes, commas and colons included: the scripts export it as `MMAE_NOTE` and pass `wandb.notes=${oc.env:MMAE_NOTE}`, and it lands in `config.yaml`, `run.json` and wandb. Set `CUDA_VISIBLE_DEVICES` to pick GPUs. The two learning rates (`train.lr` 1e-4 for new modules, `train.lr_backbone` 1e-5 for the towers) are untested placeholders.

## Evaluation

`evaluate.py` computes COCO retrieval (i2t and t2i R@1/5/10, mean and median rank, mAP, rsum = the six recalls summed).

```bash
python evaluate.py eval.run_dir=res/multimae/default/<run folder>   # a trained run: loads its config.yaml and checkpoints/best.pt, writes the metrics to its run.json
python evaluate.py model=fusion_concat                              # zero-shot pretrained CLIP, no run folder
```

`eval.split` is `test` by default (`val` is also accepted); `eval.output=<path.json>` also writes the metrics to a file. `scripts/run_eval.sh [overrides]` runs `evaluate.py data=coco_cluster` on a cluster node.

On the full COCO 5k test split, both `evaluate.py` and the test at the end of training add metrics that count more than the one paired item as correct (`eval.extended_metrics`, `mmae/engine/eccv.py`, computed with the `eccv_caption` package of Chun et al., ECCV 2022):

- `eccv/*`: ECCV Caption mAP@R, R-Precision and R@1, on its machine-and-human-verified extra positives;
- `cxc/*`: CrissCrossed Captions R@1/5/10;
- `coco1k/*`: COCO 1K R@1/5/10 (mean of 5 folds, each fold fully ranked; the ECCV Caption paper reads its R@5 and R@10 from 5K top-50 lists, which gives lower values, so only R@1 compares with it);
- `pmrp/*`: Plausible Match R-Precision, the ECCV Caption paper's version: a query's positives are its own pair and the items whose image has the same COCO object classes, and R is capped at 50. The released PM files leave the own pair out, which leaves 1,130 captions with no positive; we add it back, as the paper's Table 4 does, so all 24,760 captions and 4,952 images the files cover are queries (zero-shot CLIP ViT-B/32: 55.31, paper 55.32). It needs the PM files in `data.pm_dir` (`/data/SSD/coco/annotations/eccv_caption`, from the ECCV Caption Google Drive); null skips it.

Each comes per direction (`i2t_*`, `t2i_*`) and as the mean of the two, as in the paper's Table 4. The COCO test captions are matched to their COCO annotation ids through `captions_val2014.json` in `data.annotations_dir`.

Known-answer check: zero-shot OpenAI CLIP ViT-B/32 on the COCO 5k Karpathy test split, measured with this pipeline and published by OpenAI/CLIP:

| | ours | published |
|---|---|---|
| i2t R@1 | 50.14 | 50.1 |
| t2i R@1 | 30.44 | 30.4 |
| i2t R@5 / R@10 | 75.06 / 83.52 | |
| t2i R@5 / R@10 | 55.96 / 66.87 | |
| rsum | 361.98 | |

### VWSD

`eval.vwsd_dir` points at the SemEval-2023 Task 1 Visual Word Sense Disambiguation test package (CC-BY-NC 4.0; local `/data/SSD/vwsd`, cluster node `/local/wding/Dataset/vwsd`); null skips it. When set, `evaluate.py` adds `Hit@1` and `MRR` in percent (`mmae/engine/vwsd.py`): each item's text query is scored against its 10 candidate images. `eval.vwsd_lang` (default `en`) picks the language, `eval.vwsd_prompt` (default `{phrase}`; `{phrase}` and `{word}` are filled in) the text query. Zero-shot CLIP ViT-B/32 on the 463 English test items: Hit@1 58.10, MRR 72.79 (CPU).

```bash
python evaluate.py model=fusion_concat eval.vwsd_dir=/data/SSD/vwsd
```

### Stage 0 diagnostics

`scripts/diagnose.py` (cluster wrapper `scripts/run_diagnostics.sh`, which sets `data=coco_cluster`) encodes the COCO 5k test split with every run found under `+diag.runs_root` plus zero-shot CLIP, and writes `+diag.out/diagnostics.json`, `embeddings/` and `per_query.pt`. It reports VWSD, tower swaps (each non-contrastive run's towers are paired with the contrastive run of the same seed), class-set purity, PMRP by query type, similarity statistics and a masked-caption probe. The "class-set groups" come from ECCV Caption's PM lists, which are symmetric but not transitive on the real data, so they approximate identical COCO class sets.

```bash
scripts/run_diagnostics.sh +diag.runs_root=<dir> +diag.out=<dir> eval.vwsd_dir=/local/wding/Dataset/vwsd
```

## Known limitations

- Validation and test losses (`val/loss*`, `test/loss*`) depend on the number of GPUs. With several processes, Accelerate pads the last eval batch with duplicate pairs (`even_batches`), each rank draws its own eval masks (seed plus rank), and the contrastive negatives come from the global batch, which grows with the GPU count. Compare losses only between runs on the same number of GPUs. Retrieval metrics do not depend on it (the padded duplicates are dropped before the metrics), so neither does model selection for the two-modality configs, which monitor `val/retrieval/rsum`; `image_mae` and `text_mlm` monitor `val/loss`, which does.

## Outputs

Every training run writes one folder (Hydra writes nothing to disk):

```
res/<wandb.project>/<wandb.group or default>/<YYYYMMDD_HHMMSS>_<name>/
  config.yaml      resolved config
  run.json         status, timings, git commit, command, note, host, wandb id, best epoch, final val/test metrics
  metrics.jsonl    one line per logged step
  train.log        console log of the main process
  error.txt        traceback, only if the run failed
  checkpoints/     best.pt (train.save=best)
  plots/           learning curves
```

`python scripts/list_runs.py [--group G] [--status completed|failed|running] [--model M] [--tag T]` lists the runs. `res/` is gitignored.

## Tests

```bash
python -m pytest            # fast: CPU only, tiny random CLIP, fake COCO
python -m pytest -m slow    # real CLIP B/32, real COCO and a GPU, including the zero-shot check
```

`tests/test_model_variants.py` runs the gradient and leak-guard tests over every model switch in its `VARIANTS` dict (arm name to overrides): a new switch gets both tests by adding one entry there.

## Where the old code is

The pre-refactor code (`src/`, `main_*.py`) is frozen at the tag `legacy-v0` and the branch `legacy`.

## Docs

[`docs/reports/reports_sum.md`](docs/reports/reports_sum.md) indexes every report, starting with the [refactor report](docs/reports/auto/v1/2026-10-01_refactor.md). The design is in [`docs/superpowers/specs/2026-10-01-mmae-refactor-design.md`](docs/superpowers/specs/2026-10-01-mmae-refactor-design.md) and the implementation plan in [`docs/superpowers/plans/2026-10-01-mmae-refactor.md`](docs/superpowers/plans/2026-10-01-mmae-refactor.md).
