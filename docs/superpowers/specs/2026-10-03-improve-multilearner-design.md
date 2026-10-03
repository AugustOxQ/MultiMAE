# Improving fusion_multilearner: staged experiment design

Date: 2026-10-03. Branch `eval-baselines`. Inputs: the [baselines report](../../reports/auto/v1/2026-10-03_baselines.md),
the [lever review](../../reports/auto/v1/2026-10-03_lever_review.md) and the
[handoff](../handoffs/2026-10-03-improve-fusion-multilearner-handoff.md).

## 1. Goal

The user asked to "improve fusion_multilearner to gain more". Behind it sits the project's broad question, still
open as a topic: can masked models improve polysemy understanding in image-text retrieval? COCO is the stage where
the pipeline is made to work; other datasets come later. So this line has two aims at once: a variant that widens
multilearner's lead over the contrastive baseline, and experiments that show *why* masking moves the metrics it
moves.

Starting point (COCO 5k test, CLIP B/32, 10 epochs, batch 128, 3 seeds): multilearner against contrastive gave
ECCV mAP@R 36.95 vs 36.94 (+0.02), PMRP 56.88 vs 56.49 (+0.39, p = 0.0009), rsum 444.30 vs 441.94 (+2.36,
p = 0.11). The lever review found no prior ablation of a masked objective on ECCV Caption, PMRP, CxC or VWSD, and a
2.06 mAP@R gap between our contrastive baseline and PCME++'s InfoNCE fine-tune of the same backbone (39.0).

## 2. Decisions taken with the user (2026-10-03)

| # | Question | Decision |
|---|---|---|
| D1 | What the runs are for | **Both, in stages**: Stage 0 diagnostics, Stage 1 screening of arms that each test a mechanism and pull a lever, Stage 2 confirmation. |
| D2 | Primary metric | **ECCV Caption mAP@R** decides success. Every other metric is tracked for every run and the option to change the primary metric stays open ("keep track of others as this is just the start of the project"). |
| D3 | Recipe gap | **Hybrid**: Stage 1 masked arms run on the current recipe and reuse the existing baselines; a recipe screen runs in parallel on the contrastive model; Stage 2 runs on the improved recipe with both baselines retrained on it. |
| D4 | Fairness | Every recipe change goes to every arm, contrastive included. |
| D5 | Compute | node403 (3 A6000, reservation to about 2026-10-08 18:00) and node405 (3 GPUs, 5 days; its srun window was not yet up at 18:20 on 10-03). About 700 GPU-hours. DAS6 only, no local GPU. |

## 3. Metrics and the success bar

**Tracked for every run** (all already computed by `train.py`'s test with `eval.extended_metrics=true`): ECCV
mAP@R, R-Precision and R@1; PMRP overall, i2t and t2i; CxC R@1; COCO 1K R@1; COCO 5K recalls and rsum; validation
curves. VWSD Hit@1 and MRR are added by Stage 0 and then evaluated for every Stage 2 checkpoint. Changing the
primary metric later means re-reading `run.json`, never re-running.

**Stage 2 success bar.** A variant succeeds if, on the Stage 2 recipe and over at least 3 seeds (5 for the final
candidate if the budget allows):
1. its mean ECCV mAP@R beats both `fusion_multilearner` and `contrastive` trained on the same recipe, two-sided
   Welch p < 0.05 for each, Holm-corrected over the variants tested in Stage 2; and
2. its mean rsum is not more than 1.5 below multilearner's (about one seed std) and its mean PMRP not more than
   0.05 below.

A null on mAP@R with movement elsewhere is reported as a result, not hidden: "masking moves class-level structure
but not human-verified matches" would itself answer part of the broad question.

## 4. Hypotheses Stage 0 separates

Our result (PMRP up, mAP@R flat) has two candidate explanations from the review:
- **H1, object grounding.** Image-conditioned MLM teaches the towers which objects are present (masked content words
  need the image), and PMRP rewards shared object classes. Predicts: gains concentrated on queries whose captions
  name objects; neighbour class-set purity up in the tower(s) that carry the change; content-word masking (M2b)
  raises PMRP more than random masking at the same rate.
- **H2, softer similarity.** The auxiliary losses lower the contrastive pressure that pushes same-class items apart
  (val InfoNCE was lower for all three reconstruction models, `fusion_none` included). Predicts: similarity
  distributions with higher same-class negative similarity in every reconstruction model, `fusion_none` included,
  and no specific role for object words.

## 5. Stage 0: diagnostics on existing checkpoints (evaluation only)

On the 12 baseline checkpoints (4 models x 3 seeds) and zero-shot CLIP B/32:

| ID | Diagnostic | Output | Tests |
|---|---|---|---|
| E0a | VWSD: SemEval-2023 Task 1 English test (463 items, 10 candidate images each), Hit@1 and MRR | per model mean ± std | first reading of masked fine-tuning on lexical ambiguity |
| E0b | Tower swap: each fusion model's vision tower with the contrastive model's text tower and the reverse, same seed index | all test metrics | which tower carries the PMRP change |
| E0c | Neighbour class-set purity: share of the top-k image-image and caption-caption neighbours with the same COCO object-class set, per tower | per model | H1 |
| E0d | PMRP gain by query type: per-query PMRP difference to contrastive, split by whether the caption names a COCO class and by object count | table | H1 vs H2 |
| E0e | Similarity statistics: positive, same-class-negative and other-negative cosine distributions, learned logit scale | per model | H2 |
| E0f | Masked-caption probe: image-caption cosine as content words vs stop words are masked out | curves per model | does a masked model treat a masked caption as a less specific query |

Cost: a few GPU-hours on DAS6 (evaluation only). VWSD data goes under `/data/SSD/vwsd/` (raw dataset location rule)
and reaches the node through the cluster tool's data sync.

## 6. Stage 1: screening

**Reference arms.** Masked arms compare with the existing 3 `fusion_multilearner` runs; recipe arms with the existing
3 `contrastive` runs. All Stage 1 runs use W&B group `ml_improve` and a descriptive `wandb.name`, so they land in
`res/coco/multimae/ml_improve/`.

**Seeds.** Each arm starts with seed 42; arms get seed 43 as GPUs free up, so every arm has 2 seeds before the
advance decision. An arm that looks decisive after one seed (either way) may skip its second.

| ID | Arm | Base model | Change | Code | Status |
|---|---|---|---|---|---|
| M2a | text masking 40% | multilearner | `model.masking.text_ratio=0.4` | config | **running** (seed 42, `20261003-162109-eb85340`) |
| R3 | mean pooling | contrastive | `model.pooling=mean` | config | **running** (seed 42, `20261003-162213-eb85340`) |
| R5 | 15 epochs | contrastive | `train.epochs=15` | config | **running** (seed 42, `20261003-162314-eb85340`) |
| M5 | MAE off, image-conditioned MLM kept | multilearner | `model.loss.weights.mae=0` (the image decoder still runs; wasted compute is accepted for a screen) | config | next free GPU |
| M2b | content-word masking | multilearner | 15% of tokens drawn only from content words (stop words and punctuation never masked) | small | needs code |
| M1 | MLM reads the full image | multilearner | the text decoder's image memory comes from the clean pass's patch tokens (projected), not the 25% visible in the masked pass; fusion runs twice, (masked image, masked text) for the image decoder as today and (clean image, masked text) for the text decoder; variants with and without stop-gradient into the vision tower | small | needs code |
| M3 | reconstruction conditioned on the other modality's pooled embedding | multilearner | the clean image embedding (projected) is an extra memory token for the text decoder, the clean text embedding for the image decoder; never a modality's own clean embedding (it would leak the masked content) | small to moderate | needs code |
| M6 | masked view as a less specific positive | multilearner | an extra InfoNCE between the masked caption's pooled (EOS) embedding and the clean images, weight 0.25 | moderate | needs code |
| R2 | PCME++ learning rates | contrastive | text tower 5e-5, vision tower 5e-6, layer-wise decay 0.7, vision tower frozen for the first 2 epochs | small | needs code |

M2a dose points at 25% and 60% are added only if 40% passes the advance rule (dose-response for H1).

**Advance rule (Stage 1 to Stage 2).** With 2 seeds, compared with the reference arm's 3-seed mean:
- a masked arm advances if its mean ECCV mAP@R is at least +0.3, or its mean PMRP is at least +0.15 while its mean
  mAP@R is no more than 0.2 below; and its mean rsum is no more than 3 below. At most the two best arms advance
  (ranked by mAP@R, then PMRP).
- a recipe change is adopted for Stage 2 if it raises contrastive's mean mAP@R by at least +0.3 without lowering rsum
  by more than 3. If two or more are adopted, their combination is run once (1 seed) before Stage 2 to check they
  add up.

The +0.3 mAP@R threshold is about 1.6 standard errors of the difference (seed std 0.2), deliberately lenient so
screening rarely discards a real effect; Stage 2 carries the real test.

## 7. Stage 2: confirmation

Recipe: the current one plus the adopted recipe changes. Arms: `contrastive`, `fusion_multilearner` and the (at most
two) advanced arms, 3 seeds each (42, 43, 44), with a seeded training sampler (section 8) so seed k gives every arm
the same data order. The final candidate gets seeds 45 and 46 if the budget allows. Analysis: Welch tests on every
tracked metric against both baselines, Holm over the variants, paired tests by seed as a secondary analysis, VWSD for
every checkpoint, and the Stage 0 diagnostics rerun on the winner.

## 8. Code changes

Every change is a config switch whose default keeps today's behaviour, so existing configs and the 12 baseline runs
stay reproducible, and the existing fast tests keep passing unchanged.

| Switch (default) | Arm | Where |
|---|---|---|
| `model.masking.text_mode: random` (`content`) | M2b | `mmae/models/masking.py`, collate passes a content-token mask built from a stop-word and punctuation list over CLIP BPE tokens |
| `model.mlm_image_source: masked` (`clean`, `clean_detached`) | M1 | `mmae/models/model.py`; towers expose clean token features alongside the pooled embedding so the clean pass is not run twice |
| `model.pooled_conditioning: false` | M3 | `model.py` and `fusion.py` (extra memory token per decoder, cross-modal only) |
| `model.loss.weights.masked_view: 0.0` | M6 | `model.py`, `losses.py` (masked-caption EOS embedding vs clean images, gathered like the main InfoNCE) |
| `train.lr_text`, `train.lr_vision` (null = `lr_backbone`), `train.layer_decay: 1.0`, `train.freeze_vision_epochs: 0` | R2 | `model.param_groups`, `engine/trainer.py`; `train/debug.yaml` repeats every new train key |
| `data.seeded_sampler: false` | Stage 2 | `train.py` / data loader: the training shuffle draws from a generator seeded by `cfg.seed`, independent of model construction |
| VWSD evaluation and the Stage 0 diagnostics | Stage 0 | `mmae/engine/vwsd.py`, `mmae/engine/diagnostics.py`, `evaluate.py` switches, `scripts/run_diagnostics.sh` (the cluster tool launches only `python train.py` or `bash scripts/run_*.sh`) |

Testing (project rules): each switch gets fast CPU tests with the tiny random CLIP; every new guard is checked to
fail when the guarded code is broken on purpose; the "every trainable parameter gets a gradient" test covers the new
configs; M3's leak guard is tested directly (the text decoder never receives the clean text embedding); the tests
that pin default behaviour must pass unchanged. Edits to existing source get entries in `.claude/20261003_log.md`.

## 9. Budget (GPU-hours, one A6000 per run)

| Item | Runs | GPU-h |
|---|---|---|
| Stage 0 diagnostics | evaluation only | about 5 |
| Stage 1 masked arms (M1 x2 variants, M2a, M2b, M3, M5, M6), 2 seeds | 14 x 7.2 | about 101 |
| Stage 1 recipe arms (R2, R3, R5), 2 seeds | 6 x 4.2 to 6.3 | about 30 |
| Stage 1 extras (M2a dose points, recipe combination) | up to 5 | about 30 |
| Stage 2: 4 arms x 3 seeds (up to 1.5x longer if 15 epochs is adopted) | 12 | 60 to 130 |
| Final candidate seeds 45, 46 | 2 | up to 22 |
| Total | | about 250 to 320 of about 700 |

The rest is reserve for reruns after failures, soft-target arms (R1 in the review) or a second dose sweep. Scheduling:
config-only arms first; code arms as each passes review; node405 joins when its srun window is up.

## 10. Operations

- Launch through the cluster-run skill: commit with `cluster run` in the subject, `cluster sync`, `cluster check
  --sync-data`, `cluster launch` (one GPU per run), `cluster watch` in the background, `cluster pull` when done.
- Run registry: `tests/20261003_ml_improve/runs.md` lists every launch (tag, node, arm, seed, commit, status) and the
  pulled run folder.
- Reports: a Stage 0 report, a Stage 1 report at the advance decision and a Stage 2 report, in
  `docs/reports/auto/v1/` with rows in `reports_sum.md`, each re-deriving its numbers from the run folders.

## 11. Risks

- Masked effects on mAP@R may be below what Stage 2 can resolve (about +0.6 at 3 seeds, +0.45 at 5). The bar is
  fixed in advance; a null is reported as such.
- A Stage 1 gain on the current recipe may vanish on the improved recipe. That is the question Stage 2 asks.
- M5 runs the image decoder at weight 0: compute is wasted, behaviour is not changed.
- Mean pooling starts the retrieval projection from random weights (CLIP's pretrained projection is not used), so
  its first epochs are not comparable to native pooling's.
- node405's availability is not yet confirmed.
