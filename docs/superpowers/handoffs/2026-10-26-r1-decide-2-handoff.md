# Handoff: r1 decide 2 takes the user's own framing of the two ideas, then decision B, the grill and the spec

## Brief
- Done: step 1 (decision C), the wrap-up (34 GB freed, `hb-artelingo` merged into `main`, the reviews moved to `docs/reports/literature/`), decision A, step 3 (two literature checks, a note with three candidate ideas).
- State: no idea chosen; the user finds the three candidates interesting but framed differently from their own two ideas.
- Next: the user's own framing of idea 1 (masking under a multimodal augmentation) and idea 2 (PVSE with masked sub-spaces), then decision B, the grill and the spec.
- Blocked: nothing; the chat needs the user.

> Written 2026-10-26 04:02 by r1 decide (session d368c78b-eee4-4908-80cb-3265e80151f1). Loop step reached: after step 3, before step 4 (decision B). Direction log entry: [2026-10-26 · H-b's first readout not supported](../direction_log.md).

## 1. Read in this order

1. `docs/superpowers/direction_log.md`: "Now" and the r1 entry (the user's words on C and A, verbatim; the open B).
2. `docs/reports/auto/v1/2026-10-26_r1_candidate_ideas.md`: the four-line synthesis and the three candidates. Our recommendation sits in its collapsed block; open it only after the user's pick.
3. The two literature checks, their synthesis sections first: `docs/reports/literature/2026-10-26_masking_under_augmentation.md` (idea 1) and `docs/reports/literature/2026-10-26_masked_subspace_pvse.md` (idea 2). Each opens with a header that says how the idea was read; the user may frame it differently (section 4).
4. `docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md`, the "Results brief" section only: what the last experiment showed.
5. Before step 5: `~/.claude/references/grill.md`. Before step 6: `~/.claude/rules/spec-templates.md` and `~/.claude/templates/spec_experiment.md`.
6. `GLOSSARY.md` for the terms. `docs/reports/reports_sum.md`, "Start here", for the wider picture.

Skills: none at step 4; grilling and domain-modeling at step 5 (run as grill-with-docs, with `grill.md` overriding where they differ).

## 2. Decided by the user, not to be reopened

- **C** (2026-10-26, direction log r1 entry): the framing "can a masked loss help polysemy" is ill-defined, since a masked loss is a common auxiliary loss; masking must be used in a designed way, tied to the polysemy claim by construction.
- **A** (2026-10-26): a full loop on a new idea; no rerun of the H-b arms; step 3 was deep research on the user's two ideas plus a brainstorm (done).
- **Wrap-up** (2026-10-26): the Stage 1 screening, R2-recipe and v1-baseline checkpoints are deleted (34 GB; disk ledger); `hb-artelingo` was fast-forwarded into `main`, work goes on in `main`; the two literature reviews (2026-10-03, 2026-10-06) live in `docs/reports/literature/`; the selftest worktree was removed.
- **Not decided: B.** The user's words: "I feel all three are interesting, but also, all different from what I would frame them, so more discussion in the next chat."

## 3. Where things stand

- Last result (2026-10-08): D4 negative. JSD against dense ArtELingo-28 votes: masked model 0.277, parallel captioner 0.267, strongest linear probe 0.265, human ceiling 0.235. The only masking-specific effect: visible caption words steer the decoder (D7, log-loss 1.925 to 1.450 with 8 words), with no fair captioner comparator. Decoder reranking on COCO cut mAP@R by 15 to 19 points (D1).
- COCO (2026-10-06): 80% text masking 37.94 ECCV mAP@R over 5 seeds against 37.11 for the contrastive model. The external bar at CLIP B/32 is 40.0 to 40.2 (VSE∞, PCME++; the fair test is within our recipe).
- Step 3's four findings (the note's first section): one pooled vector cannot hold several readings for retrieval; the masked decoder is steerable by context, and its known failure is that the words alone may do the work; DML (NeurIPS 2022) shows that full caption masking with Hungarian assignment keeps 29 of 64 codes in use against 5 or 6 under partial masking, so the masking ratio is the knob; partial views carry nothing without a discriminative loss.
- The three candidates: 1 decoder-assigned image slots (about 60 GPU-hours, novelty medium, Lenses CVPR 2026 is the framing risk); 2 text-selected unmixing of blends (60 to 75 GPU-hours, novelty low, the prior that mixing moves a natural one-to-many metric is weak); 3 a context-conditioned image embedding (about 45 GPU-hours, novelty medium to high, Llip is the comparator). Details, controls and comparators in the note.
- Compute: nothing reserved; the local GPU is free; no DAS6 reservation known (run `cluster status` before a spec sets a device).
- Repo: `main`, pushed, holds everything; `hb-artelingo` is level with `main` up to this handoff; `eval-baselines` is a merged branch ref with no worktree.

## 4. Open points for the next step

- **Key, answer-first: the user's own framing of the two ideas.** How the checks read them (each check's header says so): idea 1, a masked objective under a designed multimodal augmentation, for example two images blended with each source caption a separate "either one" target, where the visible words of a masked caption select which reading to rebuild; idea 2, PVSE's K embeddings per image and caption with the K attention heads replaced by masked sub-space training, read two ways: K sub-spaces of the embedding dimensions, or K embeddings each from a different masked view, with the masked reconstruction loss as the signal that assigns readings to sub-spaces. The user said the three candidates differ from their framing: ask what differs, in their words, before anything else. If the framing differs enough, run a focused follow-up on the `deep-research` agent (Sonnet, standard depth, when the two saved checks cover the area; Opus, deep, when they do not; say which and why) rather than a new full check.
- **Key: decision B**, one of the three candidates, a combination, or the user's own design, with one line of why. The candidates not chosen go to the direction log's "Parked ideas", one line each.
- **Key, in the grill:** the masking-specific claim and its control (the words-alone control for a decoder selector; the embedding-similarity control for slots); the main metric (ECCV Caption mAP@R for one-to-many retrieval; ArtELingo for readings); the loop size (full); the device and start time (the local machine free by 11:00; no long heavy local run on weekdays 10:00 to 19:00).
- **Minor:** whether the parked collapse-correlate study (an exploration on the kept Stage 2 checkpoints) rides along.
- **At brief approval** (end of step 6): open the loop's main chat first, `loop-next ~/.claude/templates/main_chat_start.md --label "r1 main" --main --new-workspace "MultiMAE r1" --model sonnet --effort high --notify`; then the build chat with `--unattended`, after the container-restart question (`loop-chats.md`, "How to cut", step 3).

## 5. Code and pitfalls

- The repo is public: no keys, tokens or personal data in handoffs, briefs or logs. Commit with the project's git identity (project memory says how); the container has none.
- Never run `tests/20261003_ml_improve/resume.sh` or `queue_monitor.sh`: their reservation and queue folder are gone.
- The 10-07 spec names scripts never written (`scripts/hb_eval.py`, `scripts/hb_coco_probes.py`); the H-b code is `mmae/engine/hb/` and `scripts/run_hb_*.sh`.
- Retrieval today bypasses the fusion module (candidate 3 needs a conditioned pooling in the retrieval path); the blend code is the D2 probe in `mmae/engine/hb/`; no slot module (PVSE, DivE style) exists in `mmae/models/`.
- Checkpoints on disk: only the 26 Stage 2 `s2_*` COCO runs and the H-b readouts. The Stage 1, R2-recipe and baseline runs keep `run.json`, `metrics.jsonl`, `config.yaml`, `train.log` and plots, no weights.
- CLAUDE.md now says branch `main` (gitignored; the edit is logged in `.claude/20261026_log.md`). ARS is off; Codex for checks per `agent-routing.md` (`codex-usage --room` first); the `deep-research` agent for literature.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing.
- On disk over 1 GB: `res/coco/multimae/ml_improve` (18 GB, keep), `res/artelingo/hb_encoded` (1.9 GB, keep), `res/coco/diagnostics` (1.4 GB, keep); rows of 2026-10-26 in the disk ledger.
