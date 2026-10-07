"""H-b D1: decoder re-ranking of the dual encoder's top 50 captions on ECCV Caption i2t (spec 2026-10-07, section 10).

  python scripts/hb_coco_d1.py --runs res/coco/multimae/ml_improve/<run>... --out <dir> [--limit-queries N]

Per run with a caption decoder: dual embeddings of the 5k test and the first --val-images val images, top --k captions
per query, parallel and training-ratio decoder scores of every (query, candidate) pair and of each candidate with the
null image, alpha / beta tuned on val (AP@R with R = 5, the image's own captions), then ECCV mAP@R and R-Precision on
test for the dual order and re-rankers (a) and (b), overall and by thirds of R. Writes <out>/<run>/d1.json.
Runs without a decoder (contrastive, single-modality) are skipped. Scores are batch-chunk dependent only through the
mask seeds (seed + sample index), which are fixed, so all models see the same chunking for the same --batch-size."""
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
from torch.utils.data import DataLoader, Subset  # noqa: E402

from mmae.data import Collator  # noqa: E402
from mmae.data.coco import CocoRetrieval  # noqa: E402
from mmae.data.transforms import build_image_transform  # noqa: E402
from mmae.engine.hb import coco_probes as cp  # noqa: E402
from mmae.engine.hb import encode  # noqa: E402
from mmae.engine.retrieval import encode_retrieval_set  # noqa: E402
from mmae.models.backbones import processor_name  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("hb_coco_d1")
KINDS = ("parallel", "ratio")
SEED = 0


def has_caption_decoder(model) -> bool:
    return bool(model.reconstruction and model.use_image and model.use_text and not model.pooled_conditioning)


def tokens_of(collator: Collator, captions: list[str]) -> dict[str, torch.Tensor]:
    tok = collator.tokenize(captions)
    return {k: tok[k] for k in ("input_ids", "attention_mask")}


def flat_captions(dataset: CocoRetrieval) -> list[str]:
    return [c for _, captions in dataset.items for c in captions]


def score_pairs(model, dataset, query_rows: np.ndarray, cand: np.ndarray, tok, kinds, samples, batch_size, workers,
                device) -> dict[str, np.ndarray]:
    """Decoder scores (Q, k) of each query image against its k candidate captions."""
    k = cand.shape[1]
    per_chunk = max(1, batch_size // k)
    loader = DataLoader(Subset(dataset, [int(i) for i in query_rows]), batch_size=per_chunk, shuffle=False,
                        num_workers=workers, collate_fn=lambda b: torch.stack([x[0] for x in b]))
    out = {kind: np.zeros(cand.shape) for kind in KINDS if kind in kinds}
    row = 0
    for images in loader:
        n = len(images)
        rows = cand[row : row + n].reshape(-1)
        pix = images.to(device).repeat_interleave(k, dim=0)
        ctok = {key: value[rows] for key, value in tok.items()}
        for kind in out:
            s = cp.caption_scores(model, pix, ctok, kind, samples, SEED)
            out[kind][row : row + n] = s.float().cpu().numpy().reshape(n, k)
        row += n
    return out


def score_null(model, cand: np.ndarray, tok, kinds, samples, batch_size, device) -> dict[str, np.ndarray]:
    """Null-image scores (Q, k): each distinct candidate caption is scored once."""
    unique, inverse = np.unique(cand, return_inverse=True)
    out = {}
    for kind in KINDS:
        if kind not in kinds:
            continue
        scores = np.zeros(len(unique))
        for start in range(0, len(unique), batch_size):
            rows = unique[start : start + batch_size]
            ctok = {key: value[rows] for key, value in tok.items()}
            s = cp.caption_scores(model, cp.null_images(len(rows)).to(device), ctok, kind, samples, SEED)
            scores[start : start + len(rows)] = s.float().cpu().numpy()
        out[kind] = scores[inverse].reshape(cand.shape)
    return out


def dual_scores(image_emb, caption_emb, query_rows, cand) -> np.ndarray:
    image = torch.nn.functional.normalize(image_emb.float(), dim=-1)[torch.as_tensor(query_rows)]
    caption = torch.nn.functional.normalize(caption_emb.float(), dim=-1)[torch.as_tensor(cand)]
    return torch.einsum("qd,qkd->qk", image, caption).numpy()


def encode_split(model, dataset, collator, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collator.retrieval)
    image, caption = encode_retrieval_set(model, loader)
    return image, caption.reshape(-1, caption.shape[-1])


def run_one(run_dir: Path, args, device) -> dict | None:
    model, cfg = encode.load_run(run_dir, device)
    if not has_caption_decoder(model):
        log.info("skip %s: no caption decoder", run_dir.name)
        return None
    images_dir = args.images_dir or cfg.data.images_dir
    annotations_dir = args.annotations_dir or cfg.data.annotations_dir
    processor = processor_name(cfg.model.backbone)
    transform = build_image_transform(processor)
    collator = Collator(processor, int(cfg.data.max_text_len))
    t0 = time.time()

    # tune on val: the first val images, their own 5 captions as positives
    val = CocoRetrieval(images_dir, annotations_dir, "val", transform, args.val_images)
    v_img, v_cap = encode_split(model, val, collator, args.batch_size, args.workers, device)
    v_rows = np.arange(len(val))
    v_cand = cp.topk_captions(v_img, v_cap, v_rows, args.k)
    v_tok = tokens_of(collator, flat_captions(val))
    v_dec = score_pairs(model, val, v_rows, v_cand, v_tok, KINDS, args.samples, args.batch_size, args.workers, device)
    v_null = score_null(model, v_cand, v_tok, KINDS, args.samples, args.batch_size, device)
    v_dual = dual_scores(v_img, v_cap, v_rows, v_cand)
    v_pos = [np.arange(5 * i, 5 * i + 5) for i in range(len(val))]
    tuned = {kind: cp.tune(v_dual, v_dec[kind], v_null[kind], v_cand, v_pos, 5) for kind in KINDS}
    log.info("val done in %.0fs: %s", time.time() - t0, tuned)

    # test: ECCV i2t queries
    test = CocoRetrieval(images_dir, annotations_dir, "test", transform)
    q = cp.eccv_i2t(annotations_dir)
    total = len(q.R)
    n = args.limit_queries or total
    q = cp.EccvI2T(q.query_image[:n], q.positives[:n], q.R[:n])
    t_img, t_cap = encode_split(model, test, collator, args.batch_size, args.workers, device)
    full = cp.topk_captions(t_img, t_cap, q.query_image, int(q.R.max()))
    cand = full[:, : args.k]
    tok = tokens_of(collator, flat_captions(test))
    dec = score_pairs(model, test, q.query_image, cand, tok, KINDS, args.samples, args.batch_size, args.workers, device)
    null = score_null(model, cand, tok, KINDS, args.samples, args.batch_size, device)
    dual = dual_scores(t_img, t_cap, q.query_image, cand)
    result = {"dual": cp.d1_metrics(full, q)}
    for kind in KINDS:
        t = tuned[kind]
        pmi_a = cp.score_a(dec[kind], null[kind], t["alpha_a"])
        pmi_b = cp.score_b(dual, dec[kind], null[kind], t["alpha_b"], t["beta"])
        result[kind] = {
            "tuned": t,
            "rerank_a": cp.d1_metrics(cp.rerank(full, pmi_a), q),
            "rerank_b": cp.d1_metrics(cp.rerank(full, pmi_b), q),
            "raw_decoder": cp.d1_metrics(cp.rerank(full, dec[kind]), q),  # alpha 0, no dual: the uncorrected decoder
        }
    meta = {
        "run": str(run_dir), "arm": cfg.get("wandb", {}).get("name") if cfg.get("wandb") else None,
        "seed": cfg.get("seed"), "mlm_image_source": str(cfg.model.get("mlm_image_source", "masked")),
        "fusion": str(cfg.model.fusion.type), "text_ratio": float(cfg.model.masking.text_ratio),
        "k": args.k, "samples": args.samples, "val_images": args.val_images, "queries": int(len(q.R)), "queries_total": int(total),
        "limit_queries": args.limit_queries, "decoder_rows_est": int(len(q.R) * args.k * 2 * (args.samples + 1)),
        "seconds": time.time() - t0,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"meta": meta, **result}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--val-images", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=250, help="decoder rows per forward (a multiple of k is best)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-queries", type=int, default=None, help="first N ECCV queries only (smoke runs)")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for run_dir in args.runs:
        log.info("run %s on %s", run_dir.name, device)
        result = run_one(Path(run_dir), args, device)
        if result is None:
            continue
        folder = args.out / Path(run_dir).name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "d1.json").write_text(json.dumps(result, indent=1))
        log.info("wrote %s", folder / "d1.json")


if __name__ == "__main__":
    main()
