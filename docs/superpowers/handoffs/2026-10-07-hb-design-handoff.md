# Handoff: designing the first H-b experiments (after the H-b literature review)

Written 2026-10-07 00:05 before a container restart. Read this first; it points to everything else.

## The job

Continue the design discussion with the user (superpowers brainstorming, architectural path: questions, then
approaches, then a written spec in `docs/superpowers/specs/`, then writing-plans). No code changes or new cluster
launches until the user approves a written spec, except resuming the running COCO queue (below).

## Read in this order

1. `docs/user_read/2026-10-06_masking_and_polysemy.md` (the line so far, plain words).
2. `docs/reports/auto/v1/2026-10-06_masked_decoder_readings_review.md`: the H-b literature review, especially the
   Summary, section 5 (measurement toolkit), section 6 (designs D1 to D11), section 7 (proposed sequence) and
   section 8 (counter-arguments). Supporting files, including the synthesis `S_synthesis.md`, in
   `docs/reports/assets/2026-10-06_masked_decoder_readings_review/`. Final review and re-review (approved):
   `tests/20261003_ml_improve/review_2026-10-06_masked_decoder_readings_review.md`.
3. `docs/reports/auto/v1/2026-10-06_masking_stages_1_2.md` (COCO results; skim).
4. `tests/20261003_ml_improve/runs.md`, the last entries (decisions of 2026-10-06 evening).
5. Memory: `~/.claude/projects/-project-MultiAlign-MultiMAE/memory/` (research question, line decisions).

## Decisions so far (user, 2026-10-06 evening)

- Goal restated: "enable a more polysemic way of understanding through cross-modal masked reconstructions".
  Split into H-a (encoder claim: pooled retrieval embeddings keep more readings; COCO leans weakly against it) and
  H-b (model claim: the MLM decoder's own distribution holds several readings that a contrastive dual encoder
  cannot). Both are worth pursuing; start with H-b.
- ArtELingo is the main testbed, COCO checkpoints the dev bed. The user's mixing idea (blend two images 50/50;
  concatenated caption = "both at once", each source caption a separate target = "either one", the polysemy
  version) is in scope as designs D2 (zero-training probe) and D10 (training).
- The review is done and approved. Main reviewer risk: what masking adds over a plain captioner, so a parallel
  captioner (Par-cap) and later an autoregressive one (AR-cap) are required baselines, plus a prior-only model, a
  soft-label probe on CLIP features, a split-half human ceiling and per-model temperature fitting.

## The pending question (ask it again)

Scope of the first spec. Options put to the user:
1. Core (recommended): D1 decoder scoring on ECCV Caption, D2 blend probe, D4 ArtELingo triad (contrastive,
   ML-80, Par-cap; MLM target "<emotion> <explanation>", InfoNCE sees the explanation only; scored against held-out
   ArtELingo-28 vote distributions), D6 view sampling, D7 context-selected readings. D5 and D10 in a second spec
   after the first D4 seed is read.
2. The review's full sequence in one spec.
3. ArtELingo only (D4, D6, D7).
4. Mixing first (D2, then D10 on COCO).
Reading unit: the annotator's emotion label (review's recommendation; the user has not objected yet, confirm).

## Facts to keep in mind for the spec

- Code: the MLM decoder is a query decoder predicting every caption position in parallel from the fused memory
  (masked-pass image tokens + masked caption through CLIP's causal text tower). `random_token_mask` rejects a text
  ratio of 1.0 (Par-cap needs a small change). `data.max_text_len=32`: measure ArtELingo token lengths first.
- Data: ArtELingo English JSONs in `/data/PDD/artelingo/` (train 308,723 captions / 61,402 paintings; retrieval
  test split 4,975 paintings x 5 captions, mean 3.37 distinct emotions over those five); images
  `/data/PDD/wikiart_proj/wikiart` (26 GB, paths match). ArtELingo-28 public CSV
  `/data/PDD/artelingo/ArtELingo/semeval-artelingo28/` (1,658 paintings, median 108 votes, 28 languages;
  1,160 of its paintings are in ArtELingo train, so hold them out of training). Keep MultiMAE distinct from CoSiR:
  emotion is only ever predicted, never a retrieval condition.
- Compute: DAS6 node405 + node411 (6 A6000), reservation to about 2026-10-10 01:30. The review's full sequence
  needs about 126 to 155 GPU-h; the binding constraint is code and review time.

## The COCO queue

Update 2026-10-07 00:15: the container restarted and the queue was resumed with `resume.sh` (it runs detached;
check with `pgrep -af "[t]ests/20261003_ml_improve/queue.sh"`). Do not start a second copy; just arm the event
monitor below in your session and report results as they land.

Stopped at 2026-10-07 00:02 for the restart; the cluster jobs keep running. Running: `s2_r2_txt80` seeds 42-44
(node405) and `s2_txt80_mae0` seeds 42-44 (node411), due about 02:45 to 03:00. Queued: `s2_r2_contrastive` seeds
42-44. Resume with `bash tests/20261003_ml_improve/resume.sh`, then re-arm the event monitor:
`Monitor(command="bash /project/MultiAlign/MultiMAE/tests/20261003_ml_improve/queue_monitor.sh", timeout_ms=1800000)`.
Compare `s2_r2_txt80` with `s2_r2_contrastive` (does the 80% gain survive the PCME++ learning-rate recipe?) and
`s2_txt80_mae0` with `s2_txt80` seeds 42-44 (does the pixel decoder matter at 80%?); `stage2_table.py` style tests.

## Rules

Commit or push only when asked; today's reports (Stages 1 and 2, the user-read summary, the H-b review) are
uncommitted on branch `ml-improve-stages-1-2`. Times in plain Amsterdam time. Never stop processes or jobs you
did not start. When stopping the queue, do not `pkill -f` a pattern that also appears in your own command line.
