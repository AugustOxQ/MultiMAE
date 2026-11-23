# Handoff: r1 decide 4 evaluates the candidate ideas of the second research round, then brainstorms with the user and takes decision B

## Brief
- Done: second research round (10 Fable agents): six literature and data reports, four brainstorm agents, a private merge of the candidates.
- State: decision B still open; both original ideas look weak as framed (agents' estimates); several new candidate clusters exist, none evaluated yet.
- Next: ask the user the open choices, then evaluate the top candidates (novelty, idea-evaluator, devil's advocate) and brainstorm.
- Blocked: nothing; the chat needs the user.

> Written 2026-11-23 by r1 decide 3 (session 8dfc8d19-97d7-407e-85ca-f637e27ef356), cut at the context meter's note. Loop step reached: step 3 redone at the user's request, before step 4 (decision B). Direction log entry: [2026-10-26 · H-b's first readout not supported](../direction_log.md).

## 1. Read in this order

1. `.scratch/r1_round2/synthesis_candidates.md` (private, gitignored): the facts several agents agree on, and the candidate clusters C1 to C9. This is the working list for the next step.
2. `.scratch/r1_round2/digest.md`: the facts digest given to the brainstorm agents (costs, noise, assets).
3. The seven reports of 2026-11-23, read the parts you need only: `docs/reports/literature/2026-11-23_{polysemy_landscape,low_compute_claims,masking_by_design,blends_and_unmixing,masked_subspaces_and_sets}.md`, `docs/reports/auto/v1/2026-11-23_r1_{assets_and_costs,t0_checks}.md`.
4. `.scratch/r1_round2/ideas_B1_masking.md`, `ideas_B2_measure.md`, `ideas_B3_methods.md`, `ideas_B4_cross.md`: the four brainstorm files, opened per candidate.
5. `docs/superpowers/handoffs/2026-11-22-r1-decide-3-handoff.md`, section 2 (what the user decided) and section 5 (pitfalls), which still hold.
6. Before step 5: `~/.claude/references/grill.md`.

Skills: none for the evaluation step (the Ducktor skills are used by path, `~/.claude/plugins/cache/ducktor/duck/0.2.3/library/`); grilling and domain-modeling at step 5.

## 2. Decided by the user, not to be reopened

- 2026-11-23: a new round of broader deep research and idea evaluation, using Ducktor's skills or ARS, on **Fable 5.2 agents at xhigh effort**, cost no object; then a brainstorm for a real, novel, applicable idea. This is the user's standing yes for Fable in this round's research, evaluation and brainstorm agents.
- 2026-11-23: hard limit on GPU: "I don't have a lot of GPU accesses so super heavy training would not be possible, or is possible but too slow to verify the claim." Every idea carries a compute tier (T0 to T3, in the digest). The 60 GPU-hour line for T2 is an agent assumption the user has not confirmed.
- 2026-11-22 and before: see the earlier handoff's section 2 (the framing of ideas 1 and 2).

## 3. Where things stand

- Agents finished: R1 to R6b, B1 to B4 (all on disk). No idea has been evaluated with the Ducktor idea-evaluator in this round yet.
- Agents' estimates (not re-checked): idea 1 as framed, P(detectable retrieval gain) about 0.08; idea 2 as framed, median effect of decoder over similarity assignment 0.0 mAP@R (90% point +0.4) at 110 to 150 GPU-hours; our own CPU checks show little several-readings structure in COCO/ECCV Caption (about 7% of images with two or more caption groups, very sensitive to the rule) and the "visible words steer the decoder" finding is mostly the text prior.
- Theory paper (arXiv 2608.01383) was verified by a brainstorm agent (existence, date, full text read); the digest is corrected in its file. Other citations are not independently verified yet (a citation check is part of the next step).
- Compute facts: a contrastive run 4.2 h, a masked run 7.2 h on one A6000 (the 3090 about the same, one smoke measurement); three seeds detect about 0.5 mAP@R.

## 4. Open points for the next step

- **Key, ask first (they change what is worth evaluating):** (a) the question about earlier work in the "Open choices" list of `.scratch/r1_round2/synthesis_candidates.md`; (b) where unpublished ideas may live, since the repo is public (nothing new of this round has been committed); (c) the confirmed compute budget (GPU-hours per week, and whether DAS6 is expected).
- Then, wave 3 (agent work, Fable xhigh by the user's order): pick 6 to 8 candidates from the synthesis (include the strongest parts of ideas 1 and 2), one agent per candidate: fresh novelty search and the Ducktor idea-evaluator (`library/idea-evaluator/SKILL.md`, by path, inputs = the candidate's text from the files, the digest, the reports). Then ARS devil's advocate (`~/.claude/plugins/cache/academic-research-skills/academic-research-skills/3.23.0/deep-research/agents/devils_advocate_agent.md`) on the top three, and one source-verification pass over the key citations.
- Then the brainstorm with the user, and **decision B** (answer-first, with one line of why), the idea(s) not chosen go to "Parked ideas".
- **Minor:** write the direction log entry and the index rows of the new reports at the next wrap-up (nothing new of 2026-11-23 is in `reports_sum.md` yet); the user-read report for the evaluation of 2026-11-22 is still not written.

## 5. Code and pitfalls

- As in the 2026-11-22 handoff, section 5. The new CPU check scripts of the agents are copied to `.scratch/r1_round2/scripts/` (per agent); their larger outputs are in the session scratchpad of r1 decide 3 (`/tmp/claude-0/-project-MultiAlign-MultiMAE/8dfc8d19-97d7-407e-85ca-f637e27ef356/scratchpad/r1_round2/`, about 450 MB, short-lived; `r6b_checks` about 280 MB).
- The repo is public: the reports of 2026-11-23 and `.scratch/` hold unpublished idea details. Nothing of this round is committed except this handoff and the chat record; wait for the user's answer (section 4b) before committing reports that carry new ideas.
- Do not re-run the agents' CPU pilots or searches; they are in the files. Agents' numbers are from one run each.

## 6. State at handoff

- Running: nothing (all ten agents finished).
- Uncommitted: the seven reports listed in section 1.3 and nothing else; `.scratch/` is gitignored.
- On disk over 1 GB: unchanged since 2026-10-26 (`res/coco/multimae/ml_improve` 18 GB, `res/artelingo/hb_encoded` 1.9 GB, `res/coco/diagnostics` 1.4 GB; ledger rows of 2026-10-26). New small item: the session scratchpad, about 450 MB in `/tmp`.
