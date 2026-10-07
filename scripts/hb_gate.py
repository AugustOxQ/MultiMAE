"""Pre-wave-2 gate (spec section 13): check the first encoded checkpoints before the remaining GPU runs launch.

  python scripts/hb_gate.py --encoded <folder> [<folder> ...] [--annotations-dir DIR] [--json OUT]

Exit codes: 0 all gated decoder runs pass; 1 a check failed; 2 no decoder run was gated; 3 a folder errored.
Non-decoder runs are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

from mmae.engine.hb import analysis  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoded", nargs="+", required=True)
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    results, failed, errored, gated = [], False, False, 0
    for folder in args.encoded:
        try:
            enc = analysis.load_encoded(folder)
            meta = enc["meta"]
            head = f"{meta['arm']} {meta['seed']}"
            if not meta["decoder"]:
                print(f"{head} skip (no decoder)")
                results.append({"folder": str(folder), "arm": meta["arm"], "seed": meta["seed"], "skipped": True})
                continue
            res = analysis.gate(enc, analysis.prior_from(enc, args.annotations_dir))
        except Exception as e:  # a crash must not look like a gate failure
            print(f"ERROR {folder}: {type(e).__name__}: {e}")
            results.append({"folder": str(folder), "error": f"{type(e).__name__}: {e}"})
            errored = True
            continue
        gated += 1
        bad = [k for k, v in res["checks"].items() if not v]
        failed |= bool(bad)
        print(f"{head} {res['readout']} T={res['temperature']:.3f} val_nll={res['val_nll']:.4f} "
              f"prior_nll={res['prior_nll']:.4f} jsd_prior={res['jsd_to_prior']:.4f} jsd_human={res['jsd_human']:.4f} "
              f"(prior {res['prior_jsd_human']:.4f}) rho={res['entropy_spearman']:.3f} "
              f"{'FAIL [' + ', '.join(bad) + ']' if bad else 'PASS'}")
        results.append({"folder": str(folder), "arm": meta["arm"], "seed": meta["seed"], **res})
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    if gated == 0:
        print("no decoder runs gated")
    if errored:
        return 3
    if gated == 0:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
