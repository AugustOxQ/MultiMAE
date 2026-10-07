"""Write mmae/data/al28_paintings.txt: every painting of the public ArtELingo-28 CSV, sorted, one per line.
H-b holds these paintings out of ArtELingo training and validation (spec 2026-10-07, section 4)."""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=DEFAULT_CSV)
parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "mmae" / "data" / "al28_paintings.txt"))
args = parser.parse_args()
paintings = sorted(set(pd.read_csv(args.csv, usecols=["painting"])["painting"]))
Path(args.out).write_text("\n".join(paintings) + "\n")
print(f"{len(paintings)} paintings -> {args.out}")
