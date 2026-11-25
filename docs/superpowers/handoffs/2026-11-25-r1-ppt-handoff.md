# Handoff: r1 ppt writes the project's first weekly meeting report and its slides (results so far, plus the top three candidate ideas)

## Brief
- Done: the evaluation of the candidate ideas is prepared, not started; it waits for the user's go in r1 decide 4.
- State: decision B open; results so far are in the reports of 2026-10-03 to 2026-10-08, the candidates in private notes.
- Next: write the weekly report and slides the way CoSiR does, as drafts for the user.
- Blocked: nothing; the chat needs the user for the meeting date and for approval of the drafts.

> Written 2026-11-25 by r1 decide 4 (session 46405f28-64a9-4bad-9c53-96f1d704b553), a side chat opened at the user's request. r1 decide 4 stays open and is not replaced by this chat. Loop step: 4, decision B open.

The user's request, 2026-11-25, verbatim: "Not really now. First set a fresh chat r1 ppt to gather so-far meaningful results, and top-3 candidate ideas, into a weekly meeting report. Follow the same fasion&requirement of CoSiR work."

## 1. Read in this order

1. `~/.claude/references/weekly-report.md` with `~/.claude/templates/weekly_report_template.md` and `weekly_slides_template.md`: the requirements. Then `~/.claude/rules/{report-kinds,report-writing,user-read-reports}.md` and `~/.claude/references/ste-writing.md`.
2. CoSiR's latest weekly as the model for fashion and length (read-only, another project's files): `/project/CoSiR/docs/reports/weekly/2026-11-04_datasets_and_retrieval.md` and `..._slides.md`; the deck shown at the meeting is `drafts/2026-11-04_datasets_and_retrieval_slides_v5.md`. The word rules and corrections of that week: `/project/CoSiR/.scratch/stage_pages/2026-11-04_weekly_v2_changes.md` and `..._v3_changes.md`. Its content is not ours: take the form only.
3. `docs/paper/r1_ideas/notes/2026-11-25_top3_for_weekly.md` (private repo, gitignored here): the three candidates, what each slide needs, what stays out.
4. Results, newest first: `docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md`, `2026-10-06_masking_stages_1_2.md`, `2026-10-03_baselines.md`, `2026-10-03_stage0_diagnostics.md`; the readable versions `docs/user_read/2026-10-25_hb_emotion_readout.md` and `2026-10-06_masking_and_polysemy.md`; `docs/superpowers/direction_log.md` for the line and the user's decision C.

Skills: none. Subagents and Codex as `weekly-report.md`, "Process", says.

## 2. Decided by the user, not to be reopened

- 2026-11-25: the request above. The evaluation of the candidates does not start now.
- 2026-11-25: unpublished ideas live in `docs/paper/`, its own private repo. The other answers of that day are in the private notes there (`r1_ideas/notes/synthesis_candidates.md`, at the end).
- 2026-10-26 (direction log): the user reads the frame "can a masked loss help polysemy" as ill-defined. The report tells the results under the user's reading, not as support for that frame.

## 3. Where things stand

- It is the project's first weekly: `docs/reports/weekly/` and `docs/reports/pptx/` are empty, so the context opener is written from nothing.
- Results that are candidates for "meaningful", with their comparators (agent's shortlist; the numbers come from the index and the notes, take them again from the reports):
  - Text masking at 80%: ECCV Caption mAP@R 37.94 (5 seeds) against 37.11 for contrastive training (3 seeds); screen 36.95 at 15% to 37.81 at 80%.
  - Where that gain sits: the image tower, and the decoder must read the image tokens of the masked pass; no polysemy signature, and masked arms lose about 3 points on VWSD.
  - The emotion readout on ArtELingo: JSD to dense human votes 0.277 (masked model) against 0.267 (parallel captioner), 0.265 (linear probe on frozen embeddings), 0.235 (human ceiling).
  - Preliminary CPU checks of 2026-11-23 (single agents, one run each, not re-checked): only where they motivate a candidate, and marked so.
- The top three are candidates 1 to 3 of the private note, by the brainstorm agents' ranking. None is evaluated and none is chosen.

## 4. Open points for the next step

- **Key, ask first:** the meeting date and time (the title slide carries the date; CoSiR's meetings were on Wednesdays, and 2026-11-25 is one). Default: 2026-11-25. If the meeting is today, offer the slides first and the report second.
- **Key:** the shortlist of results in section 3: show it in a few lines with the outline before writing, so that the user can cut or add.
- **Key:** are the candidate ideas shown in full at the meeting, or in short form? Default: one slide each plus one comparison slide, no ranking.
- **Minor:** the direction log's "Now" paragraph is of 2026-10-26 and out of date; it is r1 decide 4's job at its wrap-up, not this chat's.

## 5. Code and pitfalls

- **This repo is public until decision B.** The drafts hold unpublished ideas: they stay untracked in `docs/reports/weekly/drafts/`, and nothing of them is committed, indexed or pushed here, also not after approval, until the repo is private or the user says otherwise. Figures of the ideas follow the same rule (`docs/reports/assets/2026-11-25_weekly/`). Add files by name; never `git add docs/`.
- The seven untracked reports of 2026-11-23 in `docs/reports/` are not this chat's: do not add, move or index them.
- Candidate text goes into no web query.
- The finding "visible words steer the decoder" (2026-10-08) now reads mostly as the text prior (preliminary check of 2026-11-23). Do not repeat the old wording alone.
- `*.pptx` is gitignored. Build the deck with `~/.claude/templates/build_weekly_slides.py <slides.md>` in this project's env (`/root/miniconda3/envs/MultiMAE/bin/python`); `python-pptx` is missing from it (checked 2026-11-25): install it there first.
- Each figure is looked at by the writer before the draft goes to the user.

## 6. State at handoff

- Running: nothing.
- Uncommitted: the seven reports of 2026-11-23 (not this chat's) and, in `docs/paper/`, nothing.
- On disk over 1 GB: unchanged since 2026-10-26 (disk ledger).
- When the drafts are approved, tell the user that r1 decide 4 waits for the go on the evaluation.
