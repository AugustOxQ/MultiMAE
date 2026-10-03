# Handoff: improve fusion_multilearner

Written 2026-10-03 (after the overnight 3-seed baselines) for a fresh agent. Read this first; it points to everything
else.

## The job

**Request (user, 2026-10-03):** "I would like to start improve fusion_multilearner to gain more."

`fusion_multilearner` is the best of the three reconstruction models on COCO 5k test, but its lead over the plain
contrastive fine-tune is small: clear on PMRP, under-powered on rsum, absent on ECCV Caption (numbers below). The job
is a variant (or a set of changes) that widens that lead.

**Not yet decided, settle with the user first** (use `superpowers:brainstorming` before any code; the user prefers
the superpowers process over CCG):
- Which metric is primary. The project's real question is polysemic retrieval (memory
  `research-question-polysemy.md`), so PMRP / ECCV Caption / CxC matter more than rsum, but COCO is still the
  "make the pipeline work" stage before other datasets.
- The success bar. A reasonable default: beat `fusion_multilearner`'s own 3-seed mean on the primary metric (Welch
  test, 3+ seeds, same protocol), without losing ECCV mAP@R, and keep a tuned contrastive baseline beside it.
- Whether any tuning (learning rates, epochs) must also be applied to the contrastive baseline, so the comparison
  stays fair (the report argues yes).

## Read in this order

1. **The baselines report** `docs/reports/auto/v1/2026-10-03_baselines.md`: Summary, §2.1 (models and gradient
   paths, Figure 1), §3 (results and tests), §4 (why the numbers moved; §4.3 is the object-presence hypothesis),
   §5 (caveats), §7 (options). Every number there was re-derived from the run folders.
2. **`CLAUDE.md`** (gitignored, local): architecture, configs, extended metrics, cluster section, testing rules.
3. **The v1 refactor report** `docs/reports/auto/v1/2026-10-01_refactor.md` and the design spec
   `docs/superpowers/specs/2026-10-01-mmae-refactor-design.md`, only as background.
4. **Memory:** `~/.claude/projects/-project-MultiAlign-MultiMAE/memory/research-question-polysemy.md`.

## What the baselines established

COCO 5k Karpathy test, CLIP ViT-B/32, 10 epochs, batch 128, one A6000 per run, mean ± std over seeds 42/43/44
(`tests/20261003_baseline_queue/summary.md`, local; `docs/reports/assets/2026-10-03_baselines/runs.csv`, tracked):

| Model | rsum | t2i R@1 | ECCV mAP@R | CxC R@1 | PMRP | PMRP t2i |
|---|---|---|---|---|---|---|
| zero-shot | 361.98 | 30.44 | 26.72 | 41.99 | 55.32 | 50.68 |
| contrastive | 441.94 ± 1.30 | 46.62 ± 0.05 | 36.94 ± 0.17 | 55.66 ± 0.42 | 56.49 ± 0.05 | 51.77 ± 0.11 |
| fusion_none | 441.07 ± 1.59 | 46.30 ± 0.24 | 36.74 ± 0.31 | 55.60 ± 0.28 | 56.61 ± 0.07 | 51.98 ± 0.10 |
| fusion_concat | 443.99 ± 0.65 | 46.83 ± 0.13 | 37.01 ± 0.20 | 56.01 ± 0.25 | 56.78 ± 0.07 | 52.27 ± 0.04 |
| fusion_multilearner | 444.30 ± 1.49 | 47.05 ± 0.13 | 36.95 ± 0.24 | 56.34 ± 0.27 | 56.88 ± 0.04 | 52.32 ± 0.04 |

- vs contrastive (Welch, n = 3): multilearner PMRP +0.39 (p = 0.0009, the only one of 33 tests that survives Holm),
  t2i R@1 +0.43 (p = 0.017), rsum +2.36 (p = 0.11 test; +3.15, p = 0.011 on val), ECCV mAP@R +0.02 (p = 0.92).
- PMRP orders contrastive < none < concat < multilearner on overall, i2t and t2i. `fusion_none` adds nothing on the
  recalls, so the gain needs the cross-modal path.
- Losses (val, epoch 10): MLM 1.623 none / 1.516 concat / 1.557 multilearner (the text decoder uses the image); MAE
  0.658 / 0.715 / 0.710 (the image decoder gains nothing from the caption).
- Every run's best epoch was the last (10); val rsum still rose 0.19 to 0.40 from epoch 9 to 10, and the cosine
  schedule ends at LR 0. Longer training is untested.
- Retrieval never uses the fusion module: it ranks by CLIP's pooled embeddings (`embed_image` / `embed_text`), so
  fusion acts only through MAE/MLM gradients into the shared towers.

## Levers (from the report's §7 and the literature scan; none chosen)

- **Loss balance.** All three weights are 1.0. MaskCLIP (CVPR 2023) found MLM weight 1 fell below plain CLIP and
  used 0.05. Try MLM/MAE weights; try MAE off with image-conditioned MLM kept (the MLM side carries the effect).
- **Feature-target MAE** instead of normalised pixels (MaskCLIP, MAMO: features > tokens > pixels for retrieval).
- **Masking ratio.** TIPS (ICLR 2025): 50% image masking slightly better than 75% for retrieval.
- **Learning rates and schedule** (1e-5 towers, 1e-4 new modules are untested placeholders; longer cosine).
- **Fusion capacity** (`fusion.learner_depth` 2, `learner_ff_dim` 512, `fusion.dim` 256, decoder depth 4).
- **Use the fusion branch at retrieval** (fusion re-ranking of the top K, or set/probabilistic embeddings): the
  largest change, closest to the polysemy question.
- **Cheap diagnostics first** (evaluation only, existing checkpoints): which tower carries the PMRP change (swap a
  fusion vision tower with the baseline text tower and back), neighbour class-set purity per tower, share of masked
  tokens that are object nouns.

## State of the code and runs

- Branch `eval-baselines`: `735216a` (contrastive model, extended metrics, W&B job tags, `scripts/run_eval.sh`),
  `88e2663` (PMRP fix), both pushed; `634ddd8` (the report) local, not pushed. `main` is still at `2056ea5`; merging
  `eval-baselines` into `main` is the user's call (ask before starting a new branch from either).
- 171 fast tests pass on CPU (`python -m pytest`).
- Run folders: `res/coco/multimae/default/<time>_<model>/` (12 runs, 7.7 GB, gitignored); zero-shot in
  `res/coco/zeroshot/clip_b32_test.json`. The seed-42 fusion runs' extended metrics are in `run.json["eval"]["test"]`
  (re-evaluated), all others in `run.json["results"]["test"]`.
- Tables and figures: `python docs/reports/assets/build_2026-10-03_baselines.py`;
  `python tests/20261003_baseline_queue/summarize.py` (mean ± std per model).

## Running experiments

- **DAS6 only** (user, 2026-10-02): no training or evaluation on the local GPU. Use the cluster-run skill
  (`~/.claude/skills/cluster-run/cluster`): `cluster sync` after committing (subject must contain `cluster run`),
  `cluster check --sync-data -- python train.py model=<m> data=coco_cluster seed=<s>`, `cluster launch -- <same>`,
  evaluation with `cluster launch -- bash scripts/run_eval.sh eval.run_dir=/local/wding/res/MultiMAE/coco/...`.
  One GPU per run; `launch` picks the lowest free GPU (do not pass `--gpu-slots` unless the user names GPUs).
- **Node:** node403, 3 × RTX A6000, reservation about 97 h at 01:44 on Oct 3 (ends around Oct 7 early morning;
  check `cluster status`). node404 belongs to another project.
- **Cost:** contrastive about 4.2 h, fusion_none / concat about 6.9 h, multilearner about 7.2 h per 10-epoch run.
- **Unattended queues:** `tests/20261003_baseline_queue/queue.sh` is a working pattern (detached with `setsid nohup`,
  pulls finished runs, launches the next, relaunches a failure once, writes a summary). Watch it with a `Monitor`
  on its log and re-arm every 30 min. Background Bash watchers die at 2 h and must not be restarted after that.
- **Shell pitfall:** the Bash tool runs zsh, which does not word-split `$var`; wrap loops in `bash <<'EOF'`.
- Commit identity: `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl"`. Commit or push only when the
  user asks (cluster sync pushes the branch, which the user has allowed for cluster runs).

## Small open items

- README, CLAUDE.md and the `mmae/engine/eccv.py` docstring give zero-shot PMRP as 55.31 (a CPU computation); the
  DAS6 GPU run gives 55.3165, i.e. 55.32.
- For a given seed, the training-data shuffle differs between model types (RNG consumed at model build), so
  seed-to-seed spread includes data order. Seeding the sampler from `cfg.seed` would remove it (it would change
  future runs only).
- `*.csv` is gitignored globally; report tables in CSV need `git add -f`.
