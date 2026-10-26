# Handoff: what next after the first H-b spec (D4 not supported)

Written 2026-10-08 17:40, after the first H-b spec was finished, reported, committed and cleaned up. Read this first;
it points to everything else.

## The job

Discuss with the user what comes next. The first pre-registered test of H-b came out negative, so this is a
direction discussion (superpowers brainstorming: questions, then approaches, then a written spec). No code changes
or launches until the user approves a written spec. Nothing is running anywhere.

## Read in this order

1. `docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md`: the Summary, section 8 (caveats) and section 9 (options
   for the second spec, none chosen). Final review and re-review: `tests/20261007_hb/review_2026-10-08_hb_emotion_readout.md`.
2. `docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md`: the user's decisions and the pre-registered rules
   (decision table at the top, section 5.1 emotion weight, section 13 timeline).
3. `tests/20261007_hb/runs.md`: run registry and timeline of the spec (last entry 17:25 is the wrap-up).
4. Background as needed: the H-b literature review `docs/reports/literature/2026-10-06_masked_decoder_readings_review.md`
   (sections 6 to 8: designs D1 to D11 and counter-arguments; D3, D5, D8 to D11 have not been run), and
   `docs/user_read/2026-10-06_masking_and_polysemy.md` for the COCO line before H-b.
5. Memory in `~/.claude/projects/-project-MultiAlign-MultiMAE/memory/` (research question, line decisions, H-b spec).

## Where things stand

- **Git:** branch `hb-artelingo` at the handoff commit, tree clean, 41 commits ahead of `main` (it contains
  `ml-improve-stages-1-2`, which is 1 commit ahead of `main`). Pushed only up to `acc25ba`; the 9 later commits are
  local. Merging or opening a PR has not been decided (the user's call).
- **Compute:** DAS6 has no reservation. At 17:20 Slurm listed no job of ours on node404, node405 or node411 (the srun
  tmux windows are still open on the head node). The reservation was due to end on 2026-10-10 01:30 but had already
  ended. A new run needs a new reservation. The local RTX 3090 was free at 17:30.
- **Checkpoints:** the 12 ArtELingo D4 checkpoints were deleted after their readouts (7.7 GB). Every readout is
  kept in `res/artelingo/hb_encoded/`, and the report build doesn't need the checkpoints. The COCO checkpoints that
  D1 and D2 used (`res/coco/multimae/ml_improve/`) are untouched.
- **Removed:** both launch worktrees (`MultiMAE-queue`, `MultiMAE-readout`) and their branches (all their commits
  are in `hb-artelingo`).

## Results in one screen

| Test | Verdict | Key numbers (baseline beside) |
|---|---|---|
| D4: decoder emotion distribution vs dense ArtELingo-28 votes (1,600 held-out paintings) | Not supported: ML-80 not better than Par-cap or the probe; worse per the painting bootstrap (Holm p ≤ 0.0034), only suggested at seed level (Holm 0.105) | JSD ML-80 0.277, Par-cap 0.267, strongest probe 0.265, prior 0.318, split-half ceiling 0.235. Entropy Spearman 0.41 / 0.51 / 0.55, ceiling 0.77 |
| D4 detail | A linear probe on *any* arm's frozen embedding (C included) matches or beats every decoder; ML-80's readout stays closest to the prior; pixel decoder irrelevant | probe JSD 0.265 for every arm; ML-80+MAE minus ML-80 +0.002 (p 0.53) |
| D6: view sampling | No view-specific signal | 16 views 0.277 beat 1 view 0.285 but not the full image 0.271; view-disagreement correlation 0.177 falls to 0.030 given readout entropy; Par-cap −0.033 |
| D7: partial captions | Visible words steer ML-80's emotion slot (masking-specific by construction); the image's contribution under partial text is not established | log-loss 1.925 → 1.450 at 8 visible tokens, prior 2.041, Par-cap flat 1.874; null image 0.261 worse than the prior |
| D1: decoder re-ranking, COCO ECCV i2t | Kill for every image-reading family | re-ranking costs 15 to 19 mAP@R vs the dual order (31.1); the full-image decoder (M1 clean) beat ML-80 (13.92 vs 13.00) |
| D2: blend probe, COCO | Kill rule floor-bound (uninformative); small masking-specific selection effect | ML-80 selection index +0.035 [+0.025, +0.045] with 3 visible words; M1 clean −0.016 |
| D2 by-product | Partial-view embeddings collapse when the vision tower meets partial views only through pixel reconstruction | view-view cosine 0.97 to 0.98 (M1 clean, text-only control, M1 detached) vs 0.43 to 0.50; the same three families lost the COCO 80% gain |

The main caveat: rsum early stopping picked epoch 2 (C s42: epoch 1) while ML-80's validation emotion loss kept
falling (s42: 1.715 at epoch 2, 1.652 at epoch 6; MLM 5.11 to 4.78 by epoch 7) and Par-cap's was flat
(1.85 to 1.87). So the decoders were lightly trained, which may have hurt ML-80 more than Par-cap. These validation
losses come from the training masking, not the readout condition.

The three COCO follow-ups from 2026-10-07 (also in this branch, `tests/20261003_ml_improve/runs.md`): on the R2
recipe the 80% masking gain shrinks from +0.75 to +0.22 ECCV mAP@R (interaction −0.54, p 0.038), and MAE off at 80%
gives 38.00, the same as with MAE (37.87), so the pixel decoder plays no part in the COCO gain.

## Options for the discussion (from report section 9; none decided)

1. **D5's 2×2 on ArtELingo:** add ML-100v (25% view, 100% text masking) and full image at 80%, to find which factor
   costs ML-80 its D4 gap. 6 runs of about 3 h.
2. **AR-cap (autoregressive captioner):** the fair baseline for D7's prefix readout and the review's third captioner.
   It needs a causal decoder (the review estimated about 120 lines plus tests) and 3 runs. Rerunning D7 would also
   need a no-image condition the decoder does not penalise (for example a mismatched painting, or image dropout in
   training).
3. **Decoder-aware checkpoint selection:** select on validation emotion NLL under the readout condition (or train the
   decoder further with the towers frozen after the rsum peak). This tests the lightly-trained caveat; ML-80 and
   Par-cap at 3 seeds are 6 runs.
4. **Stop the emotion-distribution line:** the probe sets a bar no decoder reached. The masking-specific positives
   left are visible-word steering (D7, D2), which would need AR-cap to be a claim, or a return to COCO / H-a.
5. **The collapse finding as a mechanism story for the COCO gain (H-a side):** partial views read by the caption
   decoder keep a semantic embedding; partial views trained only by pixel reconstruction collapse. This fits the
   COCO result that the 80% gain needs the caption decoder to read the masked image tokens.
6. **Not yet run from the review:** D10 (blend training, the user's mixing idea in its training form), D3, D8, D9, D11.

My view (for the discussion, not decided): option 3 is the cheapest way to settle D4's one real caveat, and the
probe result says the painting-level emotion distribution already sits in the pooled embedding. If a
decoder-aware rerun still leaves ML-80 behind Par-cap, I would close the emotion-distribution line. The only
masking-specific thread left would then be visible-word steering, tested against AR-cap. Option 5 is nearly free
(the data exist) and may matter more for a COCO paper than any further H-b arm.

## Rulings made during the spec (for the user to revisit)

Fixed before the results they affect:
- The strongest probe is chosen on post-temperature validation NLL (it changed seed 44's comparator from ML-80's probe
  to C's; rho 0.544 vs 0.560, negligible).
- D7's overall kill fires only if the image gives no benefit at all six j ≥ 2 patterns.
- The text-only control is excluded from D2's "best decoder".
- D1's kill rule (every alpha below the dual by more than the seed SD of the paired difference, no gain in the top third
  of R) was fixed after the first ML-80 checkpoints had finished; the report says so.
- D2's both-covered rule is reported as floor-bound, with the @50 ceiling beside it.

Engineering:
- Wave-2 launches and C seed 44 went ahead before the gate (C has no decoder).
- The D7 null image goes through the same views as the real image; readouts run in fp32 (both overrode the plan).
- D1's prior-corrected score is snapped to 1e-4 before ranking, and the D1/D2 tables recompute with that one scorer.
- The bootstrap raises on NaN instead of reporting p = 0.
- The AL-28 CSV is read without pandas (absent in the cluster env).
- Cluster annotations come from the hard-link folder `/data/PDD/artelingo_mmae` via a new `DATA_MAP` entry in
  `~/.claude/skills/cluster-run/cluster.conf`; `projects/MultiMAE.conf` `IMAGE_ANNOTATION_GLOB` lists the ArtELingo
  JSONs and `al28_images.json`.
- Four unpushed commit trailers written by Haiku implementers were rewritten to Opus (trees unchanged).

## Facts for any new run

- **Arms** (Hydra overrides, all with `data=artelingo_cluster train.seeded_sampler=true wandb.group=hb_d4`, seeds 42 to 44):
  - C: `model=contrastive`
  - ML-80: `model=fusion_multilearner model.masking.text_ratio=0.8 model.loss.weights.mae=0 model.emotion_head=true`
  - ML-80+MAE: the same without `model.loss.weights.mae=0`
  - Par-cap: `model=fusion_multilearner model.masking.text_ratio=1.0 model.mlm_image_source=clean model.loss.weights.mae=0 model.emotion_head=true`
  - The emotion term has weight 0.07 in every arm (`model.loss.weights.emotion`, base.yaml).
- **Pipeline:** train (`train.py`) → readout (`scripts/run_hb_encode.sh`, cluster, about 27 min per decoder run) →
  gate (`scripts/hb_gate.py`) → tables (`scripts/hb_table.py --encoded-root res/artelingo/hb_encoded --out ... --B 10000`,
  CPU about 30 min). COCO probes: `scripts/hb_coco_d1.py` (about 23 min per checkpoint, local GPU),
  `scripts/hb_coco_d2.py` (about 6 h local), `scripts/hb_coco_table.py`. The report's section 10 has the exact commands.
- **Timing:** about 3 h per decoder run (early stop with patience 5 ends near epoch 7), 1.5 h per C run.
- **Queue:** `tests/20261007_hb/queue.sh` (+ `queue_monitor.sh`, `resume.sh`), adapted from the COCO queue; it launches
  from a worktree. Readout job tags in `readout_jobs.txt`.
- **Cluster quirks:** `cluster sync` refuses unless the HEAD subject contains "cluster run", so launch from a worktree
  whose HEAD is a `cluster run: ...` commit (the last one is `388b0ce`; later commits touched only local scripts and
  docs). Node data paths: annotations `/local/wding/Dataset/artelingo_mmae`, images
  `/local/wding/Dataset/wikiart_proj/wikiart` (a new node needs `cluster sync` to copy them). Results pull to
  `res/artelingo/`.
- **Commits:** the container has no git identity; commit with
  `git -c user.name="Wangyuan Ding" -c user.email="w.ding@uva.nl" commit`. Haiku implementers write their own model
  name in the trailer: use sonnet, or fix trailers before pushing.

## Files left on disk

- `res/artelingo/hb_encoded/` 1.9 GB (all 12 readouts; inputs to every D4/D6/D7 number).
- `res/hb/gate/` 250 MB (the seed-42 gate), `res/hb/{d1,d2,d4_final,coco_tables*}` under 40 MB together,
  `res/artelingo/multimae/hb_d4/` 44 MB (run folders without checkpoints).
- `.superpowers/sdd/` (gitignored, about 1.5 MB): SDD ledgers, task briefs and review diffs of plans hb-1 to hb-3 and
  the report. Safe to delete; the report's final review is copied into `tests/20261007_hb/`.

## Opening prompt for the new chat

> Read docs/superpowers/handoffs/2026-10-08-after-hb-first-handoff.md and its reading list. The first H-b spec is
> done: D4 was not supported. I want to discuss what to do next (superpowers brainstorming, no code or launches
> until I approve a spec). Start with a short orientation: what we learned, the options, and your recommendation.
