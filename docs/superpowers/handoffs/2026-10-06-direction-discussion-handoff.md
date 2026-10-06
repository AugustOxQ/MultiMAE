# Handoff: choosing the direction after Stages 0 to 2 of the masking line

Written 2026-10-06 19:15 for a fresh agent. Read this first; it points to everything else.

## The job

The user wants to **discuss the direction choice and design ideas** with you. This is a conversation, not an
execution task: do not change code, launch cluster runs or start long jobs until the user and you have agreed on a
plan (use `superpowers:brainstorming` for the design discussion; the user prefers the superpowers process).

The project's broad question (user, 2026-10-03): "see if masked model can somehow improve polysemy understanding.
Not yet sure how this can lead to a proper topic now, but we need to work towards that." COCO R@K is only a sanity
check; the line is looking for a real polysemy topic.

## Read in this order

1. **User-read summary** `docs/user_read/2026-10-06_masking_and_polysemy.md`: the whole line in plain words, with
   the three options. The user has read it.
2. **Full report, Stages 1 and 2** `docs/reports/auto/v1/2026-10-06_masking_stages_1_2.md` (build script
   `docs/reports/assets/build_2026-10-06_masking_stages_1_2.py` prints every number; reviewed and approved, review in
   `tests/20261003_ml_improve/review_2026-10-06_masking_stages_1_2.md`).
3. **Stage 0 diagnostics** `docs/reports/auto/v1/2026-10-03_stage0_diagnostics.md`, the **literature review**
   `docs/reports/auto/v1/2026-10-03_lever_review.md` (B4 in its assets covers masking and polysemy), and the
   **deep-think note** `docs/reports/auto/v1/2026-10-05_cross_masking_polysemy.md` (directions A, B, C).
4. **Design spec** `docs/superpowers/specs/2026-10-03-improve-multilearner-design.md` (pre-registered rules),
   **run registry** `tests/20261003_ml_improve/runs.md` (every result and ruling in order), `CLAUDE.md` (gitignored,
   local: architecture, switches, cluster, testing rules).
5. **Memory** `~/.claude/projects/-project-MultiAlign-MultiMAE/memory/` (research question, line decisions).

## Where things stand

- **Result.** Hiding 80% of caption words (`s2_txt80`, `model.masking.text_ratio=0.8`) beat both baselines on the
  pre-registered 5-seed test: ECCV mAP@R 37.94 vs 37.24 (`fusion_multilearner`) and 37.11 (contrastive), Holm p
  0.003 and 0.001. On seeds 42 to 44 alone it narrowly missed (Holm 0.055 / 0.088). The gain is modest (+0.70, 6.8%
  of what contrastive fine-tuning adds).
- **Mechanism.** The gain sits in the image tower (tower swap 38.02 vs 37.12) and needs the MLM to read the
  masked-pass image tokens: `fusion_none`, M1 clean and M1 detached at 80% all lose it (3 seeds each). Why is open:
  three readings fit (report section 4.3).
- **Polysemy.** No sign: VWSD shows no gain (and the masked-model drop is unstable across stages); the stratified ECCV
  analysis (`tests/20261003_ml_improve/stratified_eccv.py`) shows the gain does not grow with the number of valid
  matches per query (intervals about +/-0.7 to 0.9 points).
- **User's views so far.** The gain is "a bit small". SemEval VWSD is "just word disambiguation, not real polysemy".
  ArtELingo (WikiArt paintings, about 5 annotators each giving an emotion and an explanation; used in the user's
  CoSiR project) is a candidate real-polysemy setting; decision held for this discussion.

## The decision on the table (the user's)

| Option | What it means |
|---|---|
| 1. ArtELingo pilot (previous agent's recommendation) | Train contrastive, fusion_multilearner and 80% on ArtELingo; test whether the gain grows with annotator disagreement (low-disagreement paintings as control); reading-coverage metrics |
| 2. Polysemy on COCO (deep-think direction A) | 2 x 2: InfoNCE vs a PCME++-style one-to-many loss, with and without 80% MLM; stratified by multiplicity; novelty risk (MACCO, MaskVLM, Verma et al.) |
| 3. Write up the controlled finding (deep-think direction C) | Masked captioning as cheap vision supervision, with controls and the polysemy null |

Smaller open questions: whether the MLM gradient into the vision tower is needed on its own; MAE at 80%; 80% on the
R2 (PCME++ learning-rate) recipe; set-of-embeddings ideas (deep-think direction B).

## Resources and constraints

- **Compute:** DAS6 node405 and node411 (3 A6000 each), reserved by the user, about 78 h left at 2026-10-06 19:13
  (to about 2026-10-10 01:30), all six GPUs free. Use the `cluster-run` skill; never hand-roll ssh. The unattended
  queue (`tests/20261003_ml_improve/queue.sh`, `resume.sh`, `queue_monitor.sh`; worktree
  `/project/MultiAlign/MultiMAE-queue`) is stopped and empty; its NODES are node405 node411. Launch commits need
  "cluster run" in the subject and must be synced to both nodes.
- **ArtELingo data:** `/data/PDD/artelingo/` (English JSON splits: train 308,723 captions over 61,402 paintings; test
  31,282 over 6,246; 82% of test paintings get at least 3 distinct emotions). WikiArt images and CoSiR's feature
  caches exist on /data (confirm paths before planning); our model fine-tunes from raw images.
- **Rules:** commit or push only when asked (identity `Wangyuan Ding <w.ding@uva.nl>`); times in plain Amsterdam time;
  reports per `~/.claude/rules/reports-layout.md` and `report-writing.md`; never stop processes or jobs you did not
  start.
