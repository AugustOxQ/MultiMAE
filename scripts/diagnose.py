"""Stage 0 diagnostics: encode every trained run under +diag.runs_root (and zero-shot CLIP) on the COCO 5k test
split and VWSD, then write tower swaps, class-set purity, similarity statistics, per-query PMRP and the
masked-caption probe to +diag.out/diagnostics.json (spec 2026-10-03, section 5).

  python scripts/diagnose.py data=coco_cluster +diag.runs_root=/local/wding/res/MultiMAE/coco/multimae/default \
      +diag.out=/local/wding/res/MultiMAE/coco/diagnostics/stage0 eval.vwsd_dir=/local/wding/Dataset/vwsd
"""
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from eccv_caption import Metrics  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mmae.data import CocoRetrieval, Collator, build_image_transform  # noqa: E402
from mmae.data.coco import retrieval_items  # noqa: E402
from mmae.engine.diagnostics import (  # noqa: E402
    class_set_groups, count_class_words, drop_words, neighbour_purity, per_query_rprecision, pmrp_rows,
    similarity_stats,
)
from mmae.engine.eccv import build_extended_metrics  # noqa: E402
from mmae.engine.retrieval import encode_retrieval_set, retrieval_metrics  # noqa: E402
from mmae.engine.vwsd import evaluate_vwsd  # noqa: E402
from mmae.models import MultiMAE  # noqa: E402
from mmae.models.backbones import processor_name  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
log = logging.getLogger("diagnose")


def find_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / "checkpoints" / "best.pt").is_file() and (p / "config.yaml").is_file())


def load_model(cfg: DictConfig, run_dir: Path | None, device: torch.device):
    """(model, model config, max_text_len, seed, model type) of a run folder, or of zero-shot CLIP for None."""
    if run_dir is None:
        model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
        return model.to(device).eval(), cfg.model, cfg.data.max_text_len, None, "zeroshot"
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    model = MultiMAE(run_cfg.model, max_text_len=run_cfg.data.max_text_len)
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")["model"])
    return model.to(device).eval(), run_cfg.model, run_cfg.data.max_text_len, int(run_cfg.seed), str(run_cfg.model.name)


@torch.no_grad()
def embed_texts(model, collator, texts: list[str], device, batch: int = 256) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), batch):
        tokens = collator.tokenize(texts[start : start + batch])
        chunks.append(model.embed_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device)).float().cpu())
    return torch.cat(chunks)


@torch.no_grad()
def probe(model, collator, items, image_emb: torch.Tensor, n_images: int, device) -> dict[str, float]:
    """Delete k content (or stop) words from each image's first caption: mean change in cosine to the own image,
    and t2i R@1 of the shortened caption among all test images."""
    rng = random.Random(0)
    out: dict[str, float] = {}
    for kind in ("content", "stop"):
        for k in (1, 2):
            rows, short, full = [], [], []
            for row, (_, captions) in enumerate(items[:n_images]):
                text = drop_words(captions[0], kind, k, rng)
                if text is not None:
                    rows.append(row), short.append(text), full.append(captions[0])
            if not rows:
                continue
            idx = torch.tensor(rows)
            short_emb, full_emb = embed_texts(model, collator, short, device), embed_texts(model, collator, full, device)
            own = image_emb[idx]
            out[f"{kind}{k}/delta_cos"] = ((short_emb * own).sum(-1) - (full_emb * own).sum(-1)).mean().item()
            out[f"{kind}{k}/t2i_R1"] = 100.0 * ((short_emb @ image_emb.T).argmax(dim=1) == idx).double().mean().item()
            out[f"{kind}{k}/n"] = float(len(rows))
    return out


def encode_run(cfg, diag, name, run_dir, items, out: Path, device) -> tuple[dict, tuple]:
    model, model_cfg, max_len, seed, kind = load_model(cfg, run_dir, device)
    processor = processor_name(model_cfg.backbone)
    transform, collator = build_image_transform(processor), Collator(processor, max_len)
    dataset = CocoRetrieval(cfg.data.images_dir, cfg.data.annotations_dir, "test", transform, cfg.data.limit_test)
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, num_workers=cfg.train.num_workers,
                        collate_fn=collator.retrieval)
    batches = ({key: value.to(device) for key, value in batch.items()} for batch in loader)
    image_emb, caption_emb = (t.cpu() for t in encode_retrieval_set(model, batches))
    info = {
        "model": kind, "seed": seed,
        "logit_scale": float(model.logit_scale.exp()) if model.logit_scale is not None else None,
        "test": retrieval_metrics(image_emb, caption_emb),
        "probe": probe(model, collator, items, image_emb, int(diag.get("probe_captions", 5000)), device),
    }
    if cfg.eval.get("vwsd_dir"):
        info["vwsd"] = evaluate_vwsd(model, cfg.eval.vwsd_dir, transform, collator, lang=cfg.eval.vwsd_lang,
                                     prompt=cfg.eval.vwsd_prompt, batch_size=cfg.train.eval_batch_size,
                                     num_workers=cfg.train.num_workers, device=device)
    torch.save({"image": image_emb.half(), "caption": caption_emb.half(), "model": kind, "seed": seed},
               out / "embeddings" / f"{name}.pt")
    log.info("%s (%s, seed %s): rsum %.2f", name, kind, seed, info["test"]["rsum"])
    return info, (image_emb, caption_emb, kind, seed)


def class_set_report(report, embeddings, extended, items, out: Path) -> None:
    """Purity, similarity statistics and per-query PMRP (needs the PM ground truth)."""
    pm_gts = Metrics(extra_file_dir=str(extended.pm_dir)).pm_gts
    ids_img, ids_cap = extended.image_ids, extended.caption_ids
    groups = class_set_groups(pm_gts, ids_img, ids_cap)
    rows = pmrp_rows(pm_gts, ids_img, ids_cap)
    k = ids_cap.shape[1]
    owners = np.repeat(np.arange(len(ids_img)), k)
    class_words = np.array([count_class_words(c) for _, captions in items for c in captions])
    per_query = {}
    for name, (img, cap, _, _) in embeddings.items():
        text = cap.reshape(-1, cap.shape[-1])
        t2i = per_query_rprecision(text[rows["t2i"][0]], img, rows["t2i"][1])
        i2t = per_query_rprecision(img[rows["i2t"][0]], text, rows["i2t"][1])
        per_query[name] = {"t2i": torch.from_numpy(t2i), "i2t": torch.from_numpy(i2t)}
        words = class_words[rows["t2i"][0]]
        by_words = {}
        for label, select in (("0", words == 0), ("1", words == 1), ("2", words == 2), ("3+", words >= 3)):
            if select.any():
                by_words[label] = 100.0 * float(t2i[select].mean())
        report["runs"][name].update(
            purity_i2i=neighbour_purity(img, groups),
            purity_t2t=neighbour_purity(text, np.repeat(groups, k), owners=owners),
            similarity=similarity_stats(img, cap, groups),
            pmrp_per_query={"t2i": 100.0 * t2i.mean(), "i2t": 100.0 * i2t.mean(), "mean": 50.0 * (t2i.mean() + i2t.mean())},
            pmrp_t2i_by_class_words=by_words,
        )
    torch.save(per_query, out / "per_query.pt")


def tower_swaps(report, embeddings, extended) -> None:
    contrastive = {seed: name for name, (_, _, kind, seed) in embeddings.items() if kind == "contrastive"}
    for name, (img, cap, kind, seed) in embeddings.items():
        if kind in ("contrastive", "zeroshot"):
            continue
        partner = contrastive.get(seed)
        if partner is None:
            log.info("no contrastive run with seed %s: no tower swap for %s", seed, name)
            continue
        partner_img, partner_cap = embeddings[partner][:2]
        for key, (i, c) in {f"{name}__img+contrastive_txt": (img, partner_cap),
                            f"contrastive_img+{name}__txt": (partner_img, cap)}.items():
            metrics = retrieval_metrics(i, c)
            if extended is not None:
                metrics.update(extended(i, c))
            report["swaps"][key] = metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    diag = cfg.get("diag") or {}
    out = Path(diag["out"])
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = retrieval_items(cfg.data.annotations_dir, "test")[: cfg.data.limit_test]
    runs = [(p.name, p) for p in find_runs(Path(diag["runs_root"]))]
    if diag.get("zeroshot", True):
        runs.insert(0, ("zeroshot", None))
    extended = build_extended_metrics(cfg, "test", two_modalities=True)
    report: dict = {"runs": {}, "swaps": {}}
    embeddings = {}
    for name, run_dir in runs:
        report["runs"][name], embeddings[name] = encode_run(cfg, diag, name, run_dir, items, out, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if extended is not None and extended.pm_dir is not None:
        class_set_report(report, embeddings, extended, items, out)
    else:
        log.info("no PM ground truth: purity, similarity statistics and per-query PMRP skipped")
    tower_swaps(report, embeddings, extended)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("wrote %s", out / "diagnostics.json")


if __name__ == "__main__":
    main()
