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

Override the keys on the command line (`data.images_dir=...`). `data=coco_cluster` points both at the cluster nodes' local disk. The val and test files carry five captions per image; they are flattened on the fly.

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
| `image_mae` | Image only: MAE on the CLIP vision tower (debugging, comparison). Monitors `val/loss`. |
| `text_mlm` | Text only: MLM on the CLIP text tower (debugging, comparison). Monitors `val/loss`. |

Launch scripts (both use `accelerate launch` on every visible GPU, or one process without a GPU):

```bash
scripts/run_local.sh "first try" model=fusion_multilearner train.epochs=20   # note, then Hydra overrides
scripts/run_cluster.sh 128 10 "baseline" model=fusion_concat                 # batch size, epochs, note, overrides
accelerate launch --num_processes 4 --multi_gpu train.py model=fusion_concat # by hand, one node
```

Set `CUDA_VISIBLE_DEVICES` to pick GPUs. The two learning rates (`train.lr` 1e-4 for new modules, `train.lr_backbone` 1e-5 for the towers) are untested placeholders.

## Evaluation

`evaluate.py` computes COCO retrieval (i2t and t2i R@1/5/10, mean and median rank, mAP, rsum = the six recalls summed).

```bash
python evaluate.py eval.run_dir=res/multimae/default/<run folder>   # a trained run: loads its config.yaml and checkpoints/best.pt, writes the metrics to its run.json
python evaluate.py model=fusion_concat                              # zero-shot pretrained CLIP, no run folder
```

`eval.split` is `test` by default (`val` is also accepted); `eval.output=<path.json>` also writes the metrics to a file.

Known-answer check: zero-shot OpenAI CLIP ViT-B/32 on the COCO 5k Karpathy test split, measured with this pipeline and published by OpenAI/CLIP:

| | ours | published |
|---|---|---|
| i2t R@1 | 50.14 | 50.1 |
| t2i R@1 | 30.44 | 30.4 |
| i2t R@5 / R@10 | 75.06 / 83.52 | |
| t2i R@5 / R@10 | 55.96 / 66.87 | |
| rsum | 361.98 | |

## Outputs

Every training run writes one folder (Hydra writes nothing to disk):

```
res/<wandb.project>/<wandb.group or default>/<YYYYMMDD_HHMMSS>_<name>/
  config.yaml      resolved config
  run.json         status, timings, git commit, command, host, wandb id, best epoch, final val/test metrics
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

## Where the old code is

The pre-refactor code (`src/`, `main_*.py`) is frozen at the tag `legacy-v0` and the branch `legacy`.

## Docs

[`docs/reports/reports_sum.md`](docs/reports/reports_sum.md) indexes every report, starting with the [refactor report](docs/reports/auto/v1/2026-10-01_refactor.md). The design is in [`docs/superpowers/specs/2026-10-01-mmae-refactor-design.md`](docs/superpowers/specs/2026-10-01-mmae-refactor-design.md) and the implementation plan in [`docs/superpowers/plans/2026-10-01-mmae-refactor.md`](docs/superpowers/plans/2026-10-01-mmae-refactor.md).
