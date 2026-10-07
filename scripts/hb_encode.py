"""H-b GPU stage: encode trained runs and save the arrays the CPU readout stage needs (spec 2026-10-07).

  python scripts/hb_encode.py --runs res/multimae/default/<run>... --out <dir> [--views 16 --d7-views 4]

Writes <out>/<run folder>/encode.npz and meta.json, and <out>/length_grid.json (the caption-length grid, shared)."""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from mmae.data.artelingo import EMOTIONS, heldout_paintings  # noqa: E402
from mmae.data.transforms import build_image_transform  # noqa: E402
from mmae.engine.hb import data as hb_data  # noqa: E402
from mmae.engine.hb import encode  # noqa: E402
from mmae.models.backbones import processor_name  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("hb_encode")


def get_length_grid(out: Path, annotations_dir, tokenizer, heldout, max_text_len) -> list[int]:
    path = out / "length_grid.json"
    key = {"annotations_dir": str(annotations_dir), "max_text_len": int(max_text_len)}
    if path.is_file():
        cached = json.loads(path.read_text())
        if cached.get("key") == key:
            return cached["lengths"]
    lengths = hb_data.length_grid(annotations_dir, tokenizer, heldout, max_text_len)
    path.write_text(json.dumps({"key": key, "lengths": lengths}))
    return lengths


def timed(label: str):
    class _T:
        def __enter__(self):
            self.t = time.time()
            log.info("%s ...", label)

        def __exit__(self, *exc):
            log.info("%s done in %.1fs", label, time.time() - self.t)

    return _T()


def check_images(images_dir, paths: list[str]) -> None:
    missing = sorted({p for p in paths if not (Path(images_dir) / p).is_file()})
    if missing:
        sys.exit(f"{len(missing)} image files missing under {images_dir}, first 5: {missing[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--al28-csv", default=hb_data.AL28_CSV)
    ap.add_argument("--min-votes", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--d7-views", type=int, default=4)
    ap.add_argument("--skip-d7", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast on CUDA (default: float32, as in training)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    for run_dir in args.runs:  # fail early, before any model is loaded
        cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
        images_dir = args.images_dir or cfg.data.images_dir
        annotations_dir = args.annotations_dir or cfg.data.annotations_dir
        heldout = heldout_paintings(cfg.data.get("heldout_file"))
        needed = hb_data.al28_targets(args.al28_csv, args.min_votes)[0].images
        needed += hb_data.val_labels(annotations_dir, heldout)[0].images
        if not args.skip_train:
            needed += hb_data.train_histograms(annotations_dir, heldout)[0].images
        if not args.skip_d7:
            needed += [c["image"] for c in hb_data.test_captions(annotations_dir)]
        check_images(images_dir, needed)

    for run_dir in args.runs:
        run_dir = Path(run_dir)
        log.info("run %s on %s", run_dir.name, device)
        model, cfg = encode.load_run(run_dir, device)
        images_dir = args.images_dir or cfg.data.images_dir
        annotations_dir = args.annotations_dir or cfg.data.annotations_dir
        max_text_len = int(cfg.data.max_text_len)
        processor = processor_name(cfg.model.backbone)
        tokenizer = AutoTokenizer.from_pretrained(processor)
        transform = build_image_transform(processor)
        heldout = heldout_paintings(cfg.data.get("heldout_file"))
        grid = get_length_grid(args.out, annotations_dir, tokenizer, heldout, max_text_len)
        hidden = encode.hidden_captions(tokenizer, grid, max_text_len)
        decoder = encode.has_decoder(model)
        arrays: dict[str, np.ndarray] = {}
        names: dict[str, list[str]] = {}

        al28, al28_counts = hb_data.al28_targets(args.al28_csv, args.min_votes)
        val, val_index, val_labels = hb_data.val_labels(annotations_dir, heldout)
        for tag, paintings in (("al28", al28), ("val", val)):
            with timed(f"{tag} paintings ({len(paintings.names)})"):
                views = encode.view_ids(len(paintings.names), args.views)
                out = encode.encode_paintings(model, paintings.images, images_dir, transform, hidden, views,
                                              args.batch_size, device, args.workers, args.bf16)
                arrays.update({f"{tag}_{k}": v for k, v in out.items()})
            names[tag] = paintings.names
        arrays.update(al28_counts=al28_counts, val_index=val_index, val_labels=val_labels)

        names["train"] = []
        if not args.skip_train:
            train, train_counts = hb_data.train_histograms(annotations_dir, heldout)
            with timed(f"train paintings ({len(train.names)})"):
                arrays["train_emb"] = encode.encode_embeddings(model, train.images, images_dir, transform,
                                                               args.batch_size, device, args.workers, args.bf16)
            arrays["train_counts"] = train_counts
            names["train"] = train.names

        arrays["prompts"] = encode.prompt_embeddings(model, tokenizer, max_text_len, device, EMOTIONS, args.bf16)
        if model.logit_scale is not None:
            arrays["logit_scale"] = np.float32(model.logit_scale.exp().item())

        if decoder and not args.skip_d7:
            captions = hb_data.test_captions(annotations_dir)
            with timed(f"D7 ({len(captions)} test captions)"):
                arrays.update(encode.encode_d7(model, captions, images_dir, transform, tokenizer, max_text_len,
                                               args.d7_views, args.batch_size, device, args.workers, args.bf16))

        folder = args.out / run_dir.name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "encode.npz", **arrays)
        weights = cfg.model.loss.weights
        meta = {
            "run": str(run_dir), "arm": OmegaConf.select(cfg, "wandb.name") or cfg.model.name,
            "seed": cfg.get("seed"), "decoder": decoder, "bf16": args.bf16,
            "mlm_image_source": str(cfg.model.get("mlm_image_source", "masked")) if decoder else None,
            "text_ratio": float(cfg.model.masking.text_ratio) if decoder else None,
            "mae_weight": float(weights.get("mae", 0.0)) if decoder else None,
            "views": args.views, "d7_views": args.d7_views, "lengths": grid,
            "al28": names["al28"], "val": names["val"], "train": names["train"],
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        log.info("wrote %s", folder)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
