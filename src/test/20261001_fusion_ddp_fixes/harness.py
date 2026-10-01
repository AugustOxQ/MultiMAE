"""Run the real fusion hook on small subsets of real COCO and record what it does.

Which code is tested is decided by PYTHONPATH (repo root for the fixed code, the
`git archive 27f7e44` copy for the baseline, see make_trees.py). Run directly for one process, or under
`torchrun --nproc_per_node 2` with --device cpu for a 2-process gloo run.

Records (rank 0 writes JSON): per-step train losses, per-epoch val losses and
val retrieval metrics, test losses and test retrieval metrics, the image/text counts
evalrank saw, and SHA-256 checksums of the trainable params at every evalrank call
and at the first test-loss batch (for the best-weights restore check).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import torch
from torch.utils.data import Subset

p = argparse.ArgumentParser()
p.add_argument("--hook", choices=["multi", "plain"], default="multi")
p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--n-train", type=int, default=64)
p.add_argument("--n-val", type=int, default=32)
p.add_argument("--n-test", type=int, default=32)
p.add_argument("--n-rval", type=int, default=41)
p.add_argument("--n-rtest", type=int, default=43)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--eval-batch-size", type=int, default=16)
p.add_argument("--num-workers", type=int, default=2)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--min-delta", type=float, default=1e-4)
p.add_argument("--save-dir", default=None)
p.add_argument("--save-interval", type=int, default=1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", required=True)
p.add_argument("--expect-src", default=None, help="assert src resolves to this dir")
p.add_argument(
    "--deterministic",
    action="store_true",
    help="torch.use_deterministic_algorithms(True) (needs CUBLAS_WORKSPACE_CONFIG=:4096:8)",
)
args = p.parse_args()

if args.deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from accelerate import Accelerator  # noqa: E402

import importlib  # noqa: E402

import src  # noqa: E402

if args.expect_src:
    _src = os.path.dirname(os.path.abspath(src.__file__))
    assert _src == os.path.abspath(args.expect_src), (_src, args.expect_src)

# src/hook/__init__.py rebinds the submodule names to the functions, so fetch modules by name
eval_mod = importlib.import_module("src.hook.eval_fusionmmae")
if args.hook == "multi":
    hook_mod = importlib.import_module("src.hook.train_fusionmmae_multi_learner")
    hook_fn = hook_mod.train_fusionmmae_multi_learner
    model_cls_name = "MultiModalFusionMAE_CLIP_MultiLearner"
else:
    hook_mod = importlib.import_module("src.hook.train_fusionmmae")
    hook_fn = hook_mod.train_fusionmmae
    model_cls_name = "MultiModalFusionMAE_CLIP"

REC = {
    "code": os.path.dirname(src.__file__),
    "args": vars(args),
    "train_steps": [],
    "val_steps": [],
    "val_retrieval": [],
    "test_retrieval": [],
    "evalrank_calls": [],
    "encode_data_calls": [],
    "test_loss_hash": None,
    "error": None,
}
STATE = {"phase": "train", "model": None}


def trainable_hash(model):
    """SHA-256 over the params that are not part of the CLIP encoders, in name order."""
    model = getattr(model, "module", model)
    h = hashlib.sha256()
    for n, prm in sorted(model.named_parameters()):
        if n.startswith(("vision_encoder.", "text_encoder.")):
            continue
        h.update(n.encode())
        h.update(prm.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ---- datasets: real COCO, deterministic strided subsets ----
_RealTrain = hook_mod.COCOImageTextDataset
_RealTest = hook_mod.MSCOCOTestDataset


def _strided(ds, n):
    step = max(1, len(ds) // n)
    return Subset(ds, list(range(0, step * n, step))[:n])


def coco_factory(*a, **kw):
    split = kw.get("split")
    ds = _RealTrain(*a, **kw)
    n = {"train": args.n_train, "val": args.n_val, "test": args.n_test}[split]
    if split == "test":
        STATE["phase"] = "test"
    return _strided(ds, n)


def retrieval_factory(*a, **kw):
    split = kw.get("split")
    ds = _RealTest(*a, **kw)
    n = args.n_rtest if split == "test" else args.n_rval
    return _strided(ds, n)


hook_mod.COCOImageTextDataset = coco_factory
hook_mod.MSCOCOTestDataset = retrieval_factory

# ---- keep a handle on the model instance ----
_RealModel = getattr(hook_mod, model_cls_name)


def model_factory(*a, **kw):
    m = _RealModel(*a, **kw)
    STATE["model"] = m
    return m


setattr(hook_mod, model_cls_name, model_factory)

# ---- checksum at the first test-loss batch ----
_real_mlm_loss = hook_mod.calculate_mlm_loss


def mlm_loss_wrapper(*a, **kw):
    if STATE["phase"] == "test" and REC["test_loss_hash"] is None:
        REC["test_loss_hash"] = trainable_hash(STATE["model"])
    return _real_mlm_loss(*a, **kw)


hook_mod.calculate_mlm_loss = mlm_loss_wrapper

# ---- evalrank / encode_data recorders ----
_real_evalrank = hook_mod.evalrank


def evalrank_wrapper(model, loader, *a, **kw):
    REC["evalrank_calls"].append(
        {"phase": STATE["phase"], "hash": trainable_hash(model)}
    )
    return _real_evalrank(model, loader, *a, **kw)


hook_mod.evalrank = evalrank_wrapper

_real_encode = eval_mod.encode_data


def encode_wrapper(*a, **kw):
    out = _real_encode(*a, **kw)
    img, txt, t2i, i2t = out
    REC["encode_data_calls"].append(
        {
            "num_images": int(img.shape[0]),
            "num_texts": int(txt.shape[0]),
            "t2i_map_len": int(t2i.shape[0]),
            "i2t_map_shape": list(i2t.shape),
            "t2i_map_head": t2i[:12].tolist(),
            "t2i_map_tail": t2i[-12:].tolist(),
        }
    )
    return out


eval_mod.encode_data = encode_wrapper

# ---- per-rank record of the val-loss averaging (fixed code only) ----
REC["val_mean_across_processes"] = []
if hasattr(hook_mod, "_mean_across_processes"):
    _real_mean = hook_mod._mean_across_processes

    def mean_wrapper(accelerator, value):
        out = _real_mean(accelerator, value)
        REC["val_mean_across_processes"].append({"local": value, "global": out})
        return out

    hook_mod._mean_across_processes = mean_wrapper


class RecLogger:
    """Stands in for wandb: records every metric the hook logs."""

    def log_metrics(self, metrics, step=None):
        keys = list(metrics)
        if any(k.startswith("train/step_") for k in keys):
            REC["train_steps"].append({"step": step, **metrics})
        elif any(k.startswith("val/step_") for k in keys):
            REC["val_steps"].append({"step": step, **metrics})
        elif keys and keys[0].startswith("val_retrieval/"):
            ep = len([r for r in REC["evalrank_calls"] if r["phase"] == "train"])
            while len(REC["val_retrieval"]) < ep:
                REC["val_retrieval"].append({})
            REC["val_retrieval"][ep - 1].update(metrics)
        elif keys and keys[0].startswith("test_retrieval/"):
            REC["test_retrieval"].append(metrics)


def main():
    accelerator = Accelerator(cpu=(args.device == "cpu"))
    REC["num_processes"] = accelerator.num_processes
    REC["distributed_type"] = str(accelerator.distributed_type)
    t0 = time.time()
    try:
        results = hook_fn(
            epochs=args.epochs,
            lr=1e-4,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=5e-2,
            seed=args.seed,
            data_root="/data/SSD/coco/images/",
            image_size=224,
            num_workers=args.num_workers,
            backbone_vision="openai/clip-vit-base-patch32",
            text_backbone="openai/clip-vit-base-patch32",
            text_max_len=32,
            proj_dim=256,
            fusion_method="concat",
            temperature=0.07,
            save_dir=args.save_dir,
            save_interval=args.save_interval,
            logger=RecLogger(),
            patience=args.patience,
            min_delta=args.min_delta,
            accelerator=accelerator,
        )
        REC["results"] = results
        m = STATE["model"]
        REC["frozen_params"] = sum(
            prm.numel() for prm in m.parameters() if not prm.requires_grad
        )
        REC["trainable_params"] = sum(
            prm.numel() for prm in m.parameters() if prm.requires_grad
        )
        REC["final_hash"] = trainable_hash(m)
    except Exception as e:  # record and re-raise so the exit code shows the crash
        REC["error"] = f"{type(e).__name__}: {e}"
        REC["traceback"] = traceback.format_exc()
        raise
    finally:
        REC["seconds"] = round(time.time() - t0, 1)
        REC["process_index"] = accelerator.process_index
        out = args.out
        if not accelerator.is_main_process:
            out = args.out.replace(".json", f".rank{accelerator.process_index}.json")
        with open(out, "w") as f:
            json.dump(REC, f, indent=1, default=str)
        print(f"[harness] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
