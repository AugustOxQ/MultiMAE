# Handoff: r1 decide 3 takes decision B on the user's two ideas, then the grill and the spec

## Brief
- Done: the user confirmed the figures of idea 1 (masked captions under blends) and idea 2 (PVSE with masked sub-spaces) as their own framing; our recommendation for B was given (idea 2, the masked decoder as the assignment signal, blends as its probe); the Ducktor idea evaluation of both ideas is written and indexed.
- State: both ideas "Accept with Revisions"; idea 2 rated the better bet by us and by the evaluation. B is not decided.
- Next: the user answers decision B, the role of idea 1 and whether to go on to the grill; then the grill (step 5) and the spec (step 6).
- Blocked: nothing; the chat needs the user.

> Written 2026-11-22 02:18 by r1 decide 2 (session 5b160409-67fc-4145-be62-772332135229), cut at the context meter's note. Loop step reached: step 4, decision B open, recommendation shown. Direction log entry: [2026-10-26 · H-b's first readout not supported](../direction_log.md).

## 1. Read in this order

1. `docs/superpowers/handoffs/2026-10-26-r1-decide-2-handoff.md`: sections 2, 3 and 5 still hold (what the user decided, where things stand, code and pitfalls). Do not repeat its step; its open point "the user's own framing" is settled (below).
2. The two figures, the user's own framing of the ideas, confirmed 2026-11-02: `docs/reports/assets/2026-10-26_r1_ideas/idea1_blend_masking.png` and `idea2_masked_subspaces.png` (built by `docs/reports/assets/build_r1_idea_figures.py`).
3. `docs/reports/auto/v1/2026-11-22_r1_ducktor_idea_evaluation.md`: the evaluation of both ideas (verdicts, fatal flaws, scores, top actions, trade-off, key judgment).
4. `docs/reports/auto/v1/2026-10-26_r1_candidate_ideas.md` and the two literature checks in `docs/reports/literature/` (2026-10-26), for comparators and risks.
5. Before step 5: `~/.claude/references/grill.md`. Before step 6: `~/.claude/rules/spec-templates.md` and `~/.claude/templates/spec_experiment.md`.

Skills: grilling and domain-modeling at step 5 (as grill-with-docs, `grill.md` overriding where they differ).

## 2. Decided by the user, not to be reopened

- C and A (2026-10-26, direction log r1 entry): the generic "can a masked loss help polysemy" frame is set aside; a full loop on a new idea with masking tied to the polysemy claim by construction.
- The framing of the two ideas (2026-11-02): the two figures are the user's ideas as they mean them. Idea 1: a pixel blend of two images, each source caption an "either one" target, 80% of the chosen caption's words hidden, the visible words select which source the decoder rebuilds, both full captions positives of the blend. Idea 2: PVSE with the K attention heads replaced by masked sub-spaces, reading (a) K learned dimension masks on the pooled embedding or reading (b) K masked patch views; the masked caption decoder assigns each caption to the sub-space whose decoder rebuilds it best (Hungarian); the contrastive loss trains the assigned pair; retrieval takes the max over k.
- The user asked for the Ducktor idea evaluation (2026-11-22); it ran on the skill's library copy, and nothing was written to `.duck/`.

## 3. Where things stand

- Our recommendation for B (given 2026-11-02, before the evaluation; the user has not answered): idea 2 with the masked decoder as the assignment signal; idea 1's blend as the probe of the trained model (per-sub-space both-covered and balance, from the D2 code), not as training data; Stage 1 screens reading (a) against learned-query slots (DivE or Llip style), each with decoder assignment against similarity assignment, K = 4, 80% masking, one seed on the development split (4 runs, about 20 GPU-hours); Stage 2 confirms the winning form at 15% and 80% masking with decoder assignment plus the similarity control at 80%, 3 seeds (9 runs, about 45 GPU-hours); probes on the checkpoints (blend per sub-space, sub-space usage, ArtELingo emotion separation); an optional arm "plus blend augmentation" (the two source captions must go to different sub-spaces). Parked by this recommendation: reading (b); idea 1 as training augmentation; the context-conditioned embedding (candidate 3) as fallback.
- The evaluation (2026-11-22) agrees on the order and adds: a MAJOR flaw shared by both, the recipe gap (our InfoNCE 37.1 against PCME++'s published 39.0 mAP@R), and for idea 2 the capacity and set-scoring confound (control: the same sub-spaces assigned by similarity; K = 1). It warns that reading (a)'s decoder, conditioned on a pooled vector only, may lose the image-token path behind the 0.8-point masking gain, and that reading (b) needs block or attentive views. Novelty: MINOR for both; Lenses (CVPR 2026) read only from its abstract. Neither idea is refuted by our data. ECCV Caption scores the COCO 5k test images, so every screen and sweep reads the validation split.
- Baselines on disk: contrastive 37.11, masked model at 80% 37.94 mAP@R (5 seeds); the external bar 40.0 to 40.2.
- Compute: nothing reserved; the local 3090 is free at night and weekends; `res-queue` governs local GPU work; DAS6 on request (the spec sets the device and start time, `run-planning.md`).

## 4. Open points for the next step

- **Key, answer-first: decision B.** Idea 2 with the masked decoder as the assignment signal (our recommendation and the evaluation's), or the user's own pick, with one line of why. Record "user" or "user, after seeing the recommendation" (the recommendation is already shown, so the latter unless the user picks differently). The candidates not chosen go to the direction log's "Parked ideas", one line each.
- **Key: idea 1's role.** Probe only (our recommendation), or also an optional training arm in Stage 2.
- **Key, in the grill:** K; the masking ratio ladder (15, 80, 100%); which forms enter the Stage 1 screen (reading (a), slots, reading (b) with block or attentive views); the main metric and the bar (ECCV mAP@R and R-Precision on the validation split for selection, the test split once); the mechanism controls (similarity assignment, K = 1, one vector of the same size for reading (a)); the device and start time (local nights end by 11:00; no long heavy local run on weekdays 10:00 to 19:00).
- **Minor:** the recipe gap to PCME++ (ignore, or add one recipe lever to every arm); whether the parked collapse-correlate study rides along; a user-read report for the evaluation (not written; the brief allowed no other files).
- **At brief approval** (end of step 6): open the loop's main chat first, `loop-next ~/.claude/templates/main_chat_start.md --label "r1 main" --main --new-workspace "MultiMAE r1" --model sonnet --effort high --notify`; then the build chat with `--unattended` after the GPU check (`loop-chats.md`, "How to cut", step 3).

## 5. Code and pitfalls

- As in the 2026-10-26 handoff, section 5. In addition: no slot, sub-space or multi-embedding module exists in `mmae/models/`; retrieval bypasses the fusion module; the blend code is the D2 probe in `mmae/engine/hb/`; `mmae/engine/eccv.py` scores the COCO 5k test split.
- The figures' build script is the source for any redraw (`build_r1_idea_figures.py`, SVG through `rsvg-convert`).
- Ducktor stays off in this project; its skill was run by path from `~/.claude/plugins/cache/ducktor/duck/0.2.3/library/idea-evaluator/`.

## 6. State at handoff

- Running: a local `http.server` on port 8765 serving the figure folder (harmless; kill it with `pkill -f 'http.server 8765'` if wanted).
- Uncommitted: nothing after this commit.
- On disk over 1 GB: unchanged since 2026-10-26 (`res/coco/multimae/ml_improve` 18 GB, `res/artelingo/hb_encoded` 1.9 GB, `res/coco/diagnostics` 1.4 GB; ledger rows of 2026-10-26).
