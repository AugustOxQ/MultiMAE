# MultiMAE constitution (placeholder)

> Version 0 · Placeholder, written 2026-10-08 · Scope: not fixed yet

The project is new and its direction is not settled, so this file holds no principles yet. Until it does:

- A spec's §7 Constitution check uses the user's global rules in `~/.claude/rules/` (baselines in `report-writing.md`,
  the final review in `final-review.md`, and the others that apply) and says so.
- Nothing in this file outranks a spec.

**When to write the real one.** Once the user confirms what the project works on: the main question, the datasets, and
which data is held for the paper. Then any agent can be asked to "write the MultiMAE constitution"; it starts from
`~/.claude/templates/spec_constitution.md` and the candidates below, and the user adopts it as version 1.

## Candidates seen so far (not binding)

Standing practices in the H-b spec (`specs/2026-10-07-hb-first-artelingo-design.md`) that may become principles:

- **Held-out paintings.** Every AL-28 painting is removed from every train and validation file, and a test asserts that
  none reaches them (H-b spec §4).
- **Equal treatment of arms.** The emotion loss has one fixed weight in every decoder arm, so no arm trains its emotion
  slot harder than the one it is compared with (H-b spec §5, amendment of 2026-10-07).
- **Pre-registered readout.** The main readout has its decision rule fixed in the spec before training (H-b spec §7).
- **Seeds and selection.** Three seeds (42 to 44); model selection on validation rsum (H-b spec §2).
