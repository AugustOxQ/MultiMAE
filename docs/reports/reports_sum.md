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
