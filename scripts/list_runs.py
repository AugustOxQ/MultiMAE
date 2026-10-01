"""List training runs under res/ by scanning their run.json files.

Usage:
  python scripts/list_runs.py [--root res] [--group G] [--status completed|failed|running] [--model M] [--tag T]
"""
import argparse
import json
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("**/run.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("res"))
    parser.add_argument("--group")
    parser.add_argument("--status")
    parser.add_argument("--model")
    parser.add_argument("--tag")
    args = parser.parse_args()
    rows = []
    for run in load_runs(args.root):
        if args.group and run.get("group") != args.group:
            continue
        if args.status and run.get("status") != args.status:
            continue
        if args.model and run.get("model") != args.model:
            continue
        if args.tag and args.tag not in run.get("tags", []):
            continue
        results = run.get("results") or {}
        val = (results.get("best_val") or {}).get("val/retrieval/rsum")
        test = (results.get("test") or {}).get("test/retrieval/rsum")
        rows.append(
            [
                run.get("created", ""), run.get("status", ""), run.get("group", ""), run.get("name", ""),
                str(results.get("best_epoch", "")), "" if val is None else f"{val:.2f}",
                "" if test is None else f"{test:.2f}", run.get("path", ""),
            ]
        )
    header = ["created", "status", "group", "name", "best_ep", "val_rsum", "test_rsum", "path"]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    for line in [header, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


if __name__ == "__main__":
    main()
