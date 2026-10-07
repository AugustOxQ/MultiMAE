"""H-b D1: decoder re-ranking of the dual encoder's top 50 captions on ECCV Caption i2t (spec 2026-10-07, section 10).

  python scripts/hb_coco_d1.py --runs res/coco/multimae/ml_improve/<run>... --out <dir> [--limit-queries N]

Per run with a caption decoder: dual embeddings of the 5k test and the first --val-images val images, top --k captions
per query, parallel and training-ratio decoder scores of every (query, candidate) pair and of each candidate with the
null image, alpha / beta tuned on val (AP@R with R = 5, the image's own captions), then ECCV mAP@R and R-Precision on
test for the dual order and re-rankers (a) and (b), overall and by thirds of R. Writes <out>/<run>/d1.json.
Runs without a decoder (contrastive, single-modality) are skipped.

Common random numbers: every draw is keyed by index, not by chunk position, so a pair's score does not depend on
--batch-size or chunk order. Token masks: one (K, N, T) table per split and model text ratio, indexed by caption id
(the same hidden set for every query, for the null image and for every model with that ratio). Views: encode.view_ids,
keyed by the query image (all candidates of a query share its K views, every model sees the same ones); the null image
uses view_ids(1)[s] for sample s (its content is constant). Outputs: <out>/<run>/d1.json (metrics, tuned values and the
full alpha / alpha x beta grids on test, overall and by R third) and scores.npz (candidates, dual, decoder and null
scores, per-query AP of each re-ranker).
"""
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
from mmae.models.fusion import NoFusion  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("hb_coco_d1")
KINDS = ("parallel", "ratio")
SEED = 0


def has_caption_decoder(model) -> bool:
    return bool(model.reconstruction and model.use_image and model.use_text and not model.pooled_conditioning)


def tokens_of(collator: Collator, captions: list[str]) -> dict[str, torch.Tensor]:
    tok = collator.tokenize(captions)
    return {k: tok[k] for k in ("input_ids", "attention_mask", "special_tokens_mask")}


def flat_captions(dataset: CocoRetrieval) -> list[str]:
    return [c for _, captions in dataset.items for c in captions]


def draws(model, tok, n_images: int, samples: int) -> dict:
    """The index-keyed draws of one split: views (K, n_images, 13) and token masks (K, n_captions, T)."""
    viewed = model.mlm_image_source == "masked" and not isinstance(model.fusion, NoFusion)
    return {
        "views": encode.view_ids(n_images, samples) if viewed else None,
        "masks": cp.token_mask_table(tok, model.text_ratio, samples, SEED),
        "null_views": encode.view_ids(1, samples)[:, 0] if viewed else None,
    }


def score_pairs(model, dataset, query_rows: np.ndarray, cand: np.ndarray, tok, kinds, samples, batch_size, workers,
                device, d: dict) -> dict[str, np.ndarray]:
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
        views = None
        if d["views"] is not None:
            views = d["views"][:, torch.as_tensor(query_rows[row : row + n])].repeat_interleave(k, dim=1)
        for kind in out:
            s = cp.caption_scores(model, pix, ctok, kind, samples, SEED, views=views,
                                  token_masks=d["masks"][:, torch.as_tensor(rows)])
            out[kind][row : row + n] = s.float().cpu().numpy().reshape(n, k)
        row += n
    return out


def score_null(model, cand: np.ndarray, tok, kinds, samples, batch_size, device, d: dict) -> dict[str, np.ndarray]:
    """Null-image scores (Q, k): each distinct candidate caption is scored once, with the same token masks (by caption
    id) as the image scores and views keyed on the sample only."""
    unique, inverse = np.unique(cand, return_inverse=True)
    out = {}
    for kind in KINDS:
        if kind not in kinds:
            continue
        scores = np.zeros(len(unique))
        for start in range(0, len(unique), batch_size):
            rows = unique[start : start + batch_size]
            ctok = {key: value[rows] for key, value in tok.items()}
            views = None if d["null_views"] is None else d["null_views"][:, None].expand(-1, len(rows), -1)
            s = cp.caption_scores(model, cp.null_images(len(rows)).to(device), ctok, kind, samples, SEED, views=views,
                                  token_masks=d["masks"][:, torch.as_tensor(rows)])
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
    on_device = ({k: v.to(device) for k, v in b.items()} for b in loader)
    image, caption = encode_retrieval_set(model, on_device)
    return image, caption.reshape(-1, caption.shape[-1])


def run_one(run_dir: Path, args, device) -> tuple[dict, dict] | None:
    """(d1.json content, scores.npz arrays), or None for a run without a caption decoder."""
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
    vd = draws(model, v_tok, len(val), args.samples)
    v_dec = score_pairs(model, val, v_rows, v_cand, v_tok, KINDS, args.samples, args.batch_size, args.workers, device, vd)
    v_null = score_null(model, v_cand, v_tok, KINDS, args.samples, args.batch_size, device, vd)
    v_dual = dual_scores(v_img, v_cap, v_rows, v_cand)
    v_pos = [np.arange(5 * i, 5 * i + 5) for i in range(len(val))]
    tuned = {kind: cp.tune(v_dual, v_dec[kind], v_null[kind], v_cand, v_pos, 5) for kind in KINDS}
    log.info("val done in %.0fs: %s", time.time() - t0, tuned)

    # test: ECCV i2t queries; the dual order runs to max(R max, k) and exactly the first k are re-ranked
    test = CocoRetrieval(images_dir, annotations_dir, "test", transform)
    q = cp.eccv_i2t(annotations_dir)
    total = len(q.R)
    n = args.limit_queries or total
    q = cp.EccvI2T(q.query_image[:n], q.positives[:n], q.R[:n])
    t_img, t_cap = encode_split(model, test, collator, args.batch_size, args.workers, device)
    depth = max(int(q.R.max()), args.k)
    full = cp.topk_captions(t_img, t_cap, q.query_image, depth)
    cand = full[:, : args.k]
    assert cand.shape[1] == args.k, (cand.shape, args.k)
    tok = tokens_of(collator, flat_captions(test))
    td = draws(model, tok, len(test), args.samples)
    dec = score_pairs(model, test, q.query_image, cand, tok, KINDS, args.samples, args.batch_size, args.workers, device, td)
    null = score_null(model, cand, tok, KINDS, args.samples, args.batch_size, device, td)
    dual = dual_scores(t_img, t_cap, q.query_image, cand)

    def ap(order):
        return 100 * cp.ap_at_r(order, q.positives, q.R)

    result = {"dual": cp.d1_metrics(full, q)}
    arrays = {"cand": cand, "dual": dual, "query_image": q.query_image, "R": q.R, "ap_dual": ap(full)}
    for kind in KINDS:
        t = tuned[kind]
        pmi_a = cp.score_a(dec[kind], null[kind], t["alpha_a"])
        pmi_b = cp.score_b(dual, dec[kind], null[kind], t["alpha_b"], t["beta"])
        grid_a = {str(a): cp.d1_metrics(cp.rerank(full, cp.score_a(dec[kind], null[kind], a)), q) for a in cp.ALPHAS}
        grid_b = {f"{a}_{b}": cp.d1_metrics(cp.rerank(full, cp.score_b(dual, dec[kind], null[kind], a, b)), q)
                  for a in cp.ALPHAS for b in cp.BETAS}
        result[kind] = {
            "tuned": t,
            "rerank_a": cp.d1_metrics(cp.rerank(full, pmi_a), q),
            "rerank_b": cp.d1_metrics(cp.rerank(full, pmi_b), q),
            "raw_decoder": cp.d1_metrics(cp.rerank(full, dec[kind]), q),  # alpha 0, no dual: the uncorrected decoder
            "grid_a": grid_a,  # test metrics at every alpha (the review's kill rule)
            "grid_b": grid_b,  # and every alpha_beta
        }
        arrays.update({f"dec_{kind}": dec[kind], f"null_{kind}": null[kind],
                       f"ap_a_{kind}": ap(cp.rerank(full, pmi_a)), f"ap_b_{kind}": ap(cp.rerank(full, pmi_b))})
    meta = {
        "run": str(run_dir), "arm": cfg.get("wandb", {}).get("name") if cfg.get("wandb") else None,
        "seed": cfg.get("seed"), "mlm_image_source": str(cfg.model.get("mlm_image_source", "masked")),
        "fusion": str(cfg.model.fusion.type), "text_ratio": float(cfg.model.masking.text_ratio),
        "k": args.k, "depth": depth, "samples": args.samples, "val_images": args.val_images,
        "queries": int(len(q.R)), "queries_total": int(total),
        "limit_queries": args.limit_queries, "decoder_rows_est": int(len(q.R) * args.k * 2 * (args.samples + 1)),
        "seconds": time.time() - t0,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"meta": meta, **result}, arrays


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
        done = run_one(Path(run_dir), args, device)
        if done is None:
            continue
        result, arrays = done
        folder = args.out / Path(run_dir).name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "d1.json").write_text(json.dumps(result, indent=1))
        np.savez_compressed(folder / "scores.npz", **arrays)
        log.info("wrote %s", folder / "d1.json")


if __name__ == "__main__":
    main()
