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

- **Latest report:** the [Stage 0 diagnostics](auto/v1/2026-10-03_stage0_diagnostics.md): six evaluation-only diagnostics (VWSD, tower swaps, class-set purity, similarity statistics, PMRP by classes named, a masked-caption probe) on the 12 baseline checkpoints, testing object grounding (H1) against softer similarity (H2) before the Stage 1 arms are read. Before it, the [lever review](auto/v1/2026-10-03_lever_review.md): an integrative literature review of masked objectives, one-to-many retrieval and polysemy, with 13 candidate levers for the remaining GPU budget. Before that, the [v1 baselines report](auto/v1/2026-10-03_baselines.md): the contrastive baseline, polysemy-aware test metrics and three seeds per model on full COCO. Earlier still, the [v1 refactor report](auto/v1/2026-10-01_refactor.md): what v0 got wrong, what v1 changed, how it was verified, the zero-shot CLIP B/32 known-answer check, the final review's fixes and a short GPU training check.
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
| 10-01 | [2026-10-01_refactor](auto/v1/2026-10-01_refactor.md) | v0 problems, the v1 `mmae` rewrite, the 136-test verification, the zero-shot CLIP B/32 check on COCO 5k (i2t R@1 50.14, t2i R@1 30.44), the final review's fixes, and a 2-epoch GPU check on 1000-image subsets (test rsum 494.96 vs zero-shot 479.78) |
| 10-03 | [2026-10-03_baselines](auto/v1/2026-10-03_baselines.md) | Full-COCO runs of `contrastive`, `fusion_none`, `fusion_concat`, `fusion_multilearner`, 3 seeds each, with ECCV Caption, CxC, COCO 1K and PMRP test metrics (zero-shot B/32 matches the ECCV Caption paper within 0.05); fusion raises PMRP over the contrastive baseline (56.88 vs 56.49, p = 0.0009), rsum only +2.4 (p = 0.11), ECCV mAP@R unchanged |
| 10-03 | [2026-10-03_lever_review](auto/v1/2026-10-03_lever_review.md) | Integrative literature review (4 themes, verified: 217 claim rows, no fabricated source) after the 3-seed baselines: no source ablates a masked objective on ECCV mAP@R, PMRP, CxC or VWSD; same-recipe loss-side changes moved mAP@R by +0.1 to +2.3 at CLIP B/32 (PCME++ +1.1, 3 runs) against our +0.02; our contrastive baseline sits 2.06 mAP@R below PCME++'s InfoNCE (36.94 vs 39.0); PMRP is weak (65.3%/56.6% pseudo-positive precision, Kendall τ 0.20 with mAP@R); 13 candidate levers, none chosen |
| 10-03 | [2026-10-03_stage0_diagnostics](auto/v1/2026-10-03_stage0_diagnostics.md) | Stage 0 of the improve-multilearner plan: evaluation-only diagnostics on the 12 baselines and zero-shot B/32. Swapped towers fall short of the full model in every reconstruction model, `fusion_none` included (full minus both swaps +0.26, +0.18, +0.29 PMRP), so the shortfall is a cost of mixing towers; with `fusion_none` as the control the fusion-specific PMRP gain sits in the image tower (+0.44, p = 0.010; +0.26, p = 0.015), which also matches the full models' rsum gain (+2.39); H1's class-word and purity predictions not confirmed at 3 seeds (bin-0 CI about ±0.5); H2 holds in relative form in every reconstruction model (positive to same-class margin −6.5% to −7.5%) but equally in `fusion_none`; VWSD Hit@1 55.00 contrastive vs 57.88 zero-shot, no masked effect |
| 10-05 | [2026-10-05_cross_masking_polysemy](auto/v1/2026-10-05_cross_masking_polysemy.md) | Deep-think note on the cross-masking line after Stage 1 and the first Stage 2 seeds: the mAP@R gain followed the MLM's share of the masked objective (80% text masking 37.87 vs multilearner 37.24 and contrastive 37.11, 3 seeds) and came with equal recall gains, so nothing yet ties it to polysemy; arguments for and against a one-to-many effect, an assessment of ArtELingo with the measurements that would make it a polysemy test, three candidate paper directions and a two-week plan |
| 10-06 | [2026-10-06_masking_stages_1_2](auto/v1/2026-10-06_masking_stages_1_2.md) | Stage 1 screen (15 arms) and Stage 2 confirmation of the improve-multilearner line, plus the 80% mechanism controls: ECCV mAP@R rose with the text-masking ratio (36.95 at 15% to 37.81 at 80%); on the seeded sampler 80% text masking reached 37.87 vs multilearner 37.24 and contrastive 37.11, missing the bar on seeds 42 to 44 (Holm p 0.055 / 0.088) and meeting it on the spec's pre-registered 5-seed final-candidate test (37.94, +0.70 vs multilearner, Holm p 0.003 / 0.001); MAE off did not replicate; the gain sits in the image tower (image swap 38.02 vs 37.12) and needs the masked-pass image tokens in the MLM's memory (three controls at 80% that each remove them, `fusion_none`, M1 clean and M1 detached, all fall to 36.98 to 37.11); VWSD 1.94 to 3.67 below contrastive in every masked arm and no growth of the gain with the number of valid ECCV matches, so on COCO it looks like generic vision supervision; R2 raised PMRP and rsum but not mAP@R |
