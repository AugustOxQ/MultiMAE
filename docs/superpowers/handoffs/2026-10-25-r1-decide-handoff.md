# Handoff: r1 decide: read the H-b results (step 1), wrap up with the setup migration, then decide the direction (step 2)

## Brief
- Done: the shared setup now fits several projects; MultiMAE was audited on 2026-10-21 (readiness note).
- State: last results 2026-10-08: H-b's first test (D4) not supported (JSD ML-80 0.277 vs Par-cap 0.267, probe 0.265).
- Next: a results brief for the 2026-10-08 report; the user's reading first (decision C); then the wrap-up and step 2.
- Blocked: nothing.

> Written 2026-10-25 12:23 by the setup chat 2 (session f68bc017), in `/project/claude-config`. Loop step reached:
> before step 1 of loop r1, MultiMAE's first loop on the current research loop. Direction log entry: none yet; this
> chat creates the log.

## 1. Read in this order

1. `docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md`: the Summary, section 8 (caveats) and section 9 (options).
   It opens with a Summary, not a results brief, so step 1 needs a brief first (section 4, item 1).
2. `~/.claude/templates/results_brief.md`; `report-kinds.md` and `user-read-reports.md` (in `~/.claude/rules/`) load
   when you open a file under `docs/reports/` or `docs/user_read/`.
3. `docs/superpowers/handoffs/2026-10-08-after-hb-first-handoff.md`: "Where things stand" and "Results in one screen"
   (10-08 state). Its "My view" paragraph is our earlier reading: it goes in the brief's collapsed block. Its process
   lines (superpowers brainstorming, the recommendation shown first) are out of date.
4. `/project/claude-config/upkeep/2026-10-21_setup_audit/multimae_readiness.md`: Q1 (gaps), Q2 (stale lines), Q3
   (what to keep), Q5 (the migration plan, in order).
5. Memory, `~/.claude/projects/-project-MultiAlign-MultiMAE/memory/` (four files): research facts; the history moves
   to the direction log at the wrap-up.

Skills: none for steps 1 and 2; `domain-modeling` for the glossary; `grilling` and `domain-modeling` at step 5.

## 2. Decided by the user, not to be reopened

- The loop label is `r1`; this chat is `r1 decide` (user, 2026-10-25, in the setup chat; recorded in
  `/project/claude-config/docs/superpowers/handoffs/2026-10-25-setup-multi-project-2-handoff.md`).
- The decisions of the first H-b spec stand as recorded in its decision table
  (`docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md`). The next direction is open.

## 3. Where things stand

- **Last results (2026-10-08):** D4 not supported: ML-80's emotion distribution is no closer to the dense ArtELingo-28
  votes than Par-cap's or a linear probe's (JSD 0.277 vs 0.267 and 0.265; prior 0.318, split-half ceiling 0.235). D1
  killed, D2 floor-bound, D6 no view-specific signal, D7 visible words steer ML-80's emotion slot. Main caveat: rsum
  early stopping picked epoch 2 while ML-80's validation emotion loss still fell. The full table is in the 10-08
  handoff.
- **No direction since:** no commit on any branch after 2026-10-08, and no MultiMAE chat.
- **Git:** branch `hb-artelingo` at `7376bd1`, tree clean, 42 commits ahead of `main`, level with its cached remote ref
  (no fetch). Whether it merges into `main` is not decided. A worktree `/root/worktrees/mmae-selftest` (branch
  `eval-baselines`) exists; whether it is still used is not known.
- **Setup:** the env `/root/miniconda3/envs/MultiMAE` (Python 3.11, torch 2.11.0+cu130) imports (checked 10-21; CUDA
  not checked). CLAUDE.md names `tests/` as the place of dated folders. `install.sh` and `sync.sh` include MultiMAE.
- **Missing** (readiness Q1): the direction log, `GLOSSARY.md`, the `docs-init` folders (literature, stage, weekly,
  pptx), `.scratch/` in `.gitignore` and `.scratch/pending_deletions.md`. `chats.tsv` and `briefs.md` start with this
  chat (`loop-next` writes them).
- **Compute:** DAS6 had no reservation on 10-08; not checked since. `res/` holds about 55 GB (ledger row: keep).

## 4. Open points for the next step

1. **Step 1, decision C (key).** Write the results brief for the 10-08 report from `results_brief.md`, at the top of
   the report, with our reading and recommendation in its collapsed block. Write a short user-read report beside it
   in `docs/user_read/` (agent default; the last results are 17 days old). Show the brief without our reading, ask
   the user's reading in a sentence or two, then open ours and say where the two differ. Record the user's reading.
2. **Wrap-up with the migration** (readiness Q5, items 2 to 8, in that order). Most items need no user decision.
   - Seed `docs/superpowers/direction_log.md` (`direction_log_entry.md`) from the reports, specs, handoffs and memory:
     past decisions with their real dates and who decided ("not recorded" when unknown), then the r1 entry with
     decision C, and a new "Now".
   - Then `docs-init`, memory cut to pointers, `GLOSSARY.md` (existing terms only; the user approves new names), the
     stale lines of Q2, `.scratch/` in `.gitignore`, and the storage summary.
   - **The two literature reviews** (`auto/v1/2026-10-03_lever_review.md`, `2026-10-06_masked_decoder_readings_review.md`)
     move to `docs/reports/literature/` only after the user's yes (minor; one line, with the link fixes it needs).
   - **The branch (key):** merge `hb-artelingo` into `main`, or keep it apart. Ask with our recommendation.
   - **The selftest worktree (minor):** ask whether it is still used.
3. **Step 2, decision A (key), in this chat:** the options are the report's section 9 (the 10-08 handoff lists six),
   plus the direction log's "Parked ideas" once it exists. Type and loop size from `spec_experiment.md`. Then steps 3
   to 6 as the loop says.
4. **The r1 main chat** does not exist yet (the project's first loop). Open it at brief approval
   (`loop-chats.md`, "The main chat", "Opening").

## 5. Code and pitfalls

- **The repo is public:** no keys, tokens or personal data in handoffs, briefs or logs.
- **Ignored instruction files:** `CLAUDE.md`, `.claude/code_notes.md` and `.claude/settings.local.json` are
  gitignored; `sync.sh` backs up the first two. Log edits to them in `.claude/yyyymmdd_log.md`.
- **Old queue scripts are history:** `tests/20261003_ml_improve/resume.sh` and `queue_monitor.sh`, and the resume
  advice in `multilearner-line-decisions.md`, need a reservation and a queue folder that no longer exist. Never run
  them on resume.
- **The 10-07 spec names scripts that were never written** (`scripts/hb_eval.py`, `scripts/hb_coco_probes.py`); the
  real entry points are `scripts/hb_encode.py`, `hb_coco_d1.py` and `hb_coco_d2.py`. Its test paragraph (emotion loss
  inside the MLM mean) is superseded by its amendment (a separate weight, 0.07).
- **Literature:** old memory says the reviews used the ARS plugin. ARS is off outside paper chats; step 3 uses the
  `deep-research` agent. `docs-check` finds literature notes by name only (`literature`, `novelty`), so it misses the
  two `*_review.md` files.

## 6. State at handoff

- Running: nothing for MultiMAE.
- Uncommitted: nothing in MultiMAE except this handoff (committed with `chats.tsv` and `briefs.md` after `loop-next`).
- On disk over 1 GB: `res/` (about 55 GB; disk ledger: keep, pruning for MultiMAE's own chat).
