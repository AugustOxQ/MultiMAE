"""H-b D2: blend probe on COCO test (spec 2026-10-07, section 11).

  python scripts/hb_coco_d2.py --runs res/coco/multimae/ml_improve/<run>... [--zero-shot] --out <dir>
                               [--d1-dir <d1 out>] [--limit-pairs N] [--device cpu] [--batch-size 250]

1,000 pairs of distinct COCO test images with unrelated captions (zero-shot CLIP B/32 caption embeddings; written once
to <out>/pairs.json, so every model reads the same pairs). Per pair and condition (pixel blends lambda A + (1 - lambda) B
on the normalised tensors, lambda 0.3..0.7; patch mixes of m of A's and 13 - m of B's patches at their own positions,
m 3..11) every model reads the same images and draws:
  dual (every model): both-covered@{5,10,20} of the top-k captions over the 25,000 test captions, and balance (A's
    share of the top 5 when A's and B's 10 captions are ranked by dual similarity);
  decoder (models with a caption decoder): balance by parallel score, both-covered@k after D1's re-ranker (a) (alpha
    tuned in D1, read from --d1-dir/<run>/d1.json, parallel score), and the selection index for cA (A's first caption)
    and cB with j in {0, 1, 2, 3} content tokens visible, on pure A, pure B and the 0.5 blend.
Models without a decoder (contrastive, zero-shot) get dual rows only.

Choices left open by the brief: the patch mix is a 13-patch view (positions drawn per pair from seed 2, the same for
every m, the first m from A); the dual encoder and masked-source decoders read the view through ids_keep, clean-source
and fusion_none decoders read the composite whose other patches are zero (the null image). Patch-mix decoder scores use
that single view (one sample). Blend decoder scores: masked-source models average K = 8 views (selection index: 4),
others the full image. Common random numbers: views are view_ids(P, K) indexed by pair, patch positions are keyed by
pair, so nothing depends on --batch-size, chunk order or the model. Re-ranking uses the dual top 50 of each blended
image; null scores come from view_ids(1, K) (as D1).

Output: <out>/<run or zeroshot>/d2.json and balance.npz (per-pair balances).
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # hb_coco_d1

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hb_coco_d1 as d1  # noqa: E402
from mmae.data import Collator  # noqa: E402
from mmae.data.coco import CocoRetrieval  # noqa: E402
from mmae.data.stopwords import content_token_table  # noqa: E402
from mmae.data.transforms import build_image_transform  # noqa: E402
from mmae.engine.hb import coco_probes as cp  # noqa: E402
from mmae.engine.hb import encode  # noqa: E402
from mmae.models import MultiMAE  # noqa: E402
from mmae.models.backbones import processor_name  # noqa: E402
from mmae.models.fusion import NoFusion  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("hb_coco_d2")
SEED = 0
PATCH_SEED = 2_000_000  # pair p draws its patch-mix positions from manual_seed(PATCH_SEED + p)
K_BLEND, K_SELECT = 8, 4
KS = (5, 10, 20)
JS = (0, 1, 2, 3)
DEPTH = 50
CONDITIONS = [("blend", lam) for lam in cp.LAMBDAS] + [("mix", m) for m in cp.MIX_M]


def cond_key(cond) -> str:
    return f"{cond[0]}_{cond[1]}"


def is_viewed(model) -> bool:
    return model.mlm_image_source == "masked" and not isinstance(model.fusion, NoFusion)


def condition_images(cond, imgs_a, imgs_b, pair_idx: np.ndarray):
    """(images (n, 3, H, W), ids_keep (n, 13) or None) of a condition for the pairs pair_idx."""
    kind, value = cond
    if kind == "blend":
        return cp.blend(imgs_a, imgs_b, value), None
    comps, ids = zip(*[cp.patch_mix(a, b, int(value), torch.Generator().manual_seed(PATCH_SEED + int(p)))
                       for a, b, p in zip(imgs_a, imgs_b, pair_idx)])
    return torch.stack(comps), torch.stack(ids)


@torch.no_grad()
def embed_captions(model, tok, batch_size, device) -> torch.Tensor:
    out = []
    for s in range(0, len(tok["input_ids"]), batch_size):
        ids = tok["input_ids"][s : s + batch_size].to(device)
        att = tok["attention_mask"][s : s + batch_size].to(device)
        out.append(model.embed_text(ids, att).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def embed_images(model, images: torch.Tensor, ids_keep: torch.Tensor | None) -> torch.Tensor:
    if ids_keep is None:
        return model.embed_image(images).float()
    return model.vision.pool(model.vision.encode(images, ids_keep)).float()


@torch.no_grad()
def score_captions(model, images, cap_rows: np.ndarray, tok, samples: int, device, views=None, hidden=None) -> np.ndarray:
    """(n, c) decoder scores of image i against its c caption rows. Parallel score without `hidden`; with hidden
    (n, c, T) the mean log p of those hidden tokens. views (samples, n, 13) are the visible patches per sample."""
    n, c = cap_rows.shape
    rows = torch.as_tensor(cap_rows.reshape(-1))
    pix = images.to(device).repeat_interleave(c, dim=0)
    ctok = {key: tok[key][rows] for key in ("input_ids", "attention_mask")}
    v = None if views is None else views.to(device).repeat_interleave(c, dim=1)
    kind, masks = "parallel", None
    if hidden is not None:
        kind, masks = "ratio", hidden.reshape(n * c, -1)[None].expand(samples, -1, -1)
    s = cp.caption_scores(model, pix, ctok, kind, samples, SEED, views=v, token_masks=masks)
    return s.float().cpu().numpy().reshape(n, c)


def dual_rows(img_emb, cap_emb, cap_rows: np.ndarray) -> np.ndarray:
    image = torch.nn.functional.normalize(img_emb.float(), dim=-1)
    caption = torch.nn.functional.normalize(cap_emb.float(), dim=-1)[torch.as_tensor(cap_rows)]
    return torch.einsum("pd,pcd->pc", image, caption).numpy()


def summarise(x: np.ndarray, dropped: int) -> dict:
    return {"mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
            "n": int(len(x)), "dropped": dropped}


def build_shared(dataset, collator, pairs: np.ndarray, limit: int | None) -> dict:
    """Everything every model reads: tokens, pair images, views and content table. Views are drawn for all pairs of
    the file and then sliced by --limit-pairs, so a limit changes nothing about the retained pairs."""
    tok = d1.tokens_of(collator, d1.flat_captions(dataset))
    n_all = len(pairs)
    use = pairs[:limit] if limit else pairs
    imgs = lambda col: torch.stack([dataset[int(i)][0] for i in use[:, col]])  # noqa: E731
    return {
        "tok": tok, "pairs": use,
        "a_rows": 5 * use[:, :1] + np.arange(5), "b_rows": 5 * use[:, 1:] + np.arange(5),
        "imgs_a": imgs(0), "imgs_b": imgs(1),
        "views_blend": encode.view_ids(n_all, K_BLEND)[:, : len(use)],
        "views_select": encode.view_ids(n_all, K_SELECT)[:, : len(use)],
        "null_views": encode.view_ids(1, K_BLEND)[:, 0],
        "content": content_token_table(collator.tokenizer),
    }


def run_model(model, name: str, sh: dict, args, device, d1_json: dict | None = None) -> tuple[dict, dict]:
    t0 = time.time()
    tok, P, bs = sh["tok"], len(sh["pairs"]), args.batch_size
    decoder = d1.has_caption_decoder(model)
    viewed = is_viewed(model) if decoder else False
    cap_emb = embed_captions(model, tok, bs, device)
    a_rows, b_rows = sh["a_rows"], sh["b_rows"]
    ab_rows = np.concatenate([a_rows, b_rows], axis=1)
    pb = max(1, bs // 10)  # pairs per chunk (10 captions each)
    alpha = None
    if decoder and d1_json is not None:
        alpha = d1_json["parallel"]["tuned"]["alpha_a"]

    result = {"dual": {}, "meta": {
        "run": name, "pairs": P, "decoder": decoder, "alpha_a_parallel": alpha, "viewed": viewed,
        "k_blend": K_BLEND, "k_select": K_SELECT}}
    arrays, tops, scores_dual = {}, {}, {}
    # --- dual readouts and the dual top 50 of each condition
    for cond in CONDITIONS:
        embs, sims = [], []
        for s in range(0, P, pb):
            idx = np.arange(s, min(P, s + pb))
            images, ids = condition_images(cond, sh["imgs_a"][idx], sh["imgs_b"][idx], idx)
            with torch.no_grad():
                emb = embed_images(model, images.to(device), None if ids is None else ids.to(device)).cpu()
            embs.append(emb)
            sims.append(dual_rows(emb, cap_emb, ab_rows[idx]))
        emb = torch.cat(embs)
        top = cp.topk_captions(emb, cap_emb, np.arange(P), DEPTH)
        sims = np.concatenate(sims)
        bal = cp.balance(sims[:, :5], sims[:, 5:])
        tops[cond_key(cond)] = top
        arrays[f"dual_balance_{cond_key(cond)}"] = bal
        result["dual"][cond_key(cond)] = {
            "both_covered": {str(k): cp.both_covered(top[:, :k], a_rows, b_rows) for k in KS},
            "balance": float(bal.mean())}
    if not decoder:
        result["meta"]["seconds"] = time.time() - t0
        return result, arrays

    # --- decoder: balance by parallel score
    result["decoder"] = {"balance_parallel": {}, "both_covered_rerank": {}, "selection": {}}

    def views_for(cond, idx, ids):
        if cond[0] == "mix":
            return (ids[None] if viewed else None), 1
        return (sh["views_blend"][:, idx] if viewed else None), (K_BLEND if viewed else 1)

    for cond in CONDITIONS:
        out = []
        for s in range(0, P, pb):
            idx = np.arange(s, min(P, s + pb))
            images, ids = condition_images(cond, sh["imgs_a"][idx], sh["imgs_b"][idx], idx)
            views, samples = views_for(cond, idx, ids)
            out.append(score_captions(model, images, ab_rows[idx], tok, samples, device, views))
        sc = np.concatenate(out)
        bal = cp.balance(sc[:, :5], sc[:, 5:])
        arrays[f"dec_balance_{cond_key(cond)}"] = bal
        result["decoder"]["balance_parallel"][cond_key(cond)] = float(bal.mean())

    # --- decoder: both-covered after D1's re-ranker (a), alpha from D1
    if alpha is not None:
        uniq = np.unique(np.concatenate([t.reshape(-1) for t in tops.values()]))
        null = {}
        nv = sh["null_views"]
        for s in range(0, len(uniq), bs):
            rows = uniq[s : s + bs]
            views = nv[:, None].expand(-1, len(rows), -1) if viewed else None
            sc = score_captions(model, cp.null_images(len(rows)), rows[:, None], tok, K_BLEND if viewed else 1,
                                device, views)
            null.update(zip(rows.tolist(), sc[:, 0]))
        pc = max(1, bs // DEPTH)
        for cond in CONDITIONS:
            top = tops[cond_key(cond)]
            dec = []
            for s in range(0, P, pc):
                idx = np.arange(s, min(P, s + pc))
                images, ids = condition_images(cond, sh["imgs_a"][idx], sh["imgs_b"][idx], idx)
                views, samples = views_for(cond, idx, ids)
                dec.append(score_captions(model, images, top[idx], tok, samples, device, views))
            dec = np.concatenate(dec)
            nul = np.vectorize(null.__getitem__)(top)
            order = cp.rerank(top, cp.score_a(dec, nul, alpha))
            result["decoder"]["both_covered_rerank"][cond_key(cond)] = {
                str(k): cp.both_covered(order[:, :k], a_rows, b_rows) for k in KS}

    # --- decoder: selection index, cA with A's content tokens visible and cB with B's
    ids_all, att_all = tok["input_ids"], tok["attention_mask"]
    real_all = cp._real_tokens(att_all)
    content = sh["content"][ids_all] & real_all  # (N, T)
    sel = {j: {"A": ([], [], []), "B": ([], [], [])} for j in JS}
    for s in range(0, P, pb):
        idx = np.arange(s, min(P, s + pb))
        a, b = sh["imgs_a"][idx], sh["imgs_b"][idx]
        imgs = {"A": a, "B": b, "blend": cp.blend(a, b, 0.5)}
        views = sh["views_select"][:, idx] if viewed else None
        samples = K_SELECT if viewed else 1
        for side, rows in (("A", 5 * sh["pairs"][idx, 0]), ("B", 5 * sh["pairs"][idx, 1])):
            r = torch.as_tensor(rows)
            first = torch.cumsum(content[r].int(), dim=1)
            for j in JS:
                visible = content[r] & (first <= j)
                hidden = (real_all[r] & ~visible)[:, None]
                s_img = {key: score_captions(model, im, rows[:, None], tok, samples, device, views, hidden)[:, 0]
                         for key, im in imgs.items()}
                own, other = ("A", "B") if side == "A" else ("B", "A")
                for store, val in zip(sel[j][side], (s_img["blend"], s_img[own], s_img[other])):
                    store.append(val)
    for j in JS:
        entry = {}
        for side in ("A", "B"):
            blend_, own, other = (np.concatenate(x) for x in sel[j][side])
            idx_, dropped = cp.selection_index(blend_, own, other)
            entry[f"c{side}"] = summarise(idx_, dropped)
            entry[f"c{side}"]["s_own_minus_other"] = float((own - other).mean())
        both = np.concatenate([
            cp.selection_index(*[np.concatenate(x) for x in sel[j][side]])[0] for side in ("A", "B")])
        entry["pooled"] = summarise(both, entry["cA"]["dropped"] + entry["cB"]["dropped"])
        result["decoder"]["selection"][str(j)] = entry
    result["meta"]["seconds"] = time.time() - t0
    return result, arrays


def zero_shot_model(device):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=["model=contrastive"])
    return MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len).to(device).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], type=Path)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--d1-dir", type=Path, default=None, help="D1 output folder (<run>/d1.json gives alpha)")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=250, help="decoder rows per forward")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    models = [(Path(r).name, r) for r in args.runs] + ([("zeroshot", None)] if args.zero_shot else [])
    pairs_path = args.out / "pairs.json"
    dataset = collator = None

    def setup(cfg):
        nonlocal dataset, collator
        if dataset is None:
            processor = processor_name(cfg.model.backbone)
            ann = args.annotations_dir or cfg.data.annotations_dir
            dataset = CocoRetrieval(args.images_dir or cfg.data.images_dir, ann, "test",
                                    build_image_transform(processor))
            collator = Collator(processor, int(cfg.data.max_text_len))

    if not pairs_path.exists():
        zs, cfg = zero_shot_model(device)
        setup(cfg)
        tok = d1.tokens_of(collator, d1.flat_captions(dataset))
        emb = embed_captions(zs, tok, args.batch_size, device).reshape(len(dataset), 5, -1)
        pairs = cp.select_pairs(emb)
        pairs_path.write_text(json.dumps(pairs.tolist()))
        del zs
    pairs = np.array(json.loads(pairs_path.read_text()))
    shared = None
    for name, run in models:
        if run is None:
            model, cfg = zero_shot_model(device)
        else:
            model, cfg = encode.load_run(run, device)
        setup(cfg)
        shared = shared or build_shared(dataset, collator, pairs, args.limit_pairs)
        d1_json = None
        if args.d1_dir is not None and (args.d1_dir / name / "d1.json").exists():
            d1_json = json.loads((args.d1_dir / name / "d1.json").read_text())
        log.info("model %s on %s (d1 alpha: %s)", name, device, d1_json is not None)
        result, arrays = run_model(model, name, shared, args, device, d1_json)
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "d2.json").write_text(json.dumps(result, indent=1))
        np.savez_compressed(folder / "balance.npz", **arrays)
        log.info("wrote %s", folder / "d2.json")
        del model


if __name__ == "__main__":
    main()
