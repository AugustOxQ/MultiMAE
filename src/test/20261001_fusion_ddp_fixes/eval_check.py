"""evalrank on a fixed model and a fixed retrieval subset, with 1 or 2 processes.

The model is MultiModalFusionMAE_CLIP_MultiLearner built with a fixed seed (untrained
heads on top of pretrained CLIP), so every run sees the same weights. Its retrieval is
near chance; --model clip swaps its encode_*_tokens_cls for pretrained CLIP's projected
features so that wrong index maps show up as a large drop in recall. Which code is
tested is decided by PYTHONPATH; --expect-src asserts it.

--mode accel    : Accelerator(cpu=...), model and loader prepared, evalrank(..., accelerator)
--mode noaccel  : plain model and DataLoader, evalrank(model, loader) (the test-script path)

--unprepared    : accel mode only: prepare the model but pass the raw DataLoader (fixed code must
                  raise ValueError under >1 process; the baseline silently double-counts)

--variant (baseline probes, accel mode only):
  as_is         : pass the prepared (DDP-wrapped under 2 processes) model, as the hooks do
  unwrap        : pass the unwrapped model, to get past the DDP AttributeError
  unwrap_cuda   : also turn Tensor.cuda() into a no-op, to get past the hard-coded .cuda()
"""
import argparse
import importlib
import json
import os
import sys
import traceback

import torch
from torch.utils.data import DataLoader, Subset

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["accel", "noaccel"], default="accel")
p.add_argument("--variant", choices=["as_is", "unwrap", "unwrap_cuda"], default="as_is")
p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
p.add_argument(
    "--model",
    choices=["fusion", "clip"],
    default="fusion",
    help="clip: route encode_*_tokens_cls through pretrained CLIP projected features "
    "so retrieval has real signal (the fusion model still gets wrapped/unwrapped)",
)
p.add_argument("--unprepared", action="store_true")
p.add_argument("--n", type=int, default=41, help="strided subset size; 0 = full split")
p.add_argument("--bs", type=int, default=16)
p.add_argument("--num-workers", type=int, default=2)
p.add_argument("--expect-src", default=None)
p.add_argument("--out", required=True)
args = p.parse_args()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import src  # noqa: E402

SRC = os.path.dirname(os.path.abspath(src.__file__))
if args.expect_src:
    assert SRC == os.path.abspath(args.expect_src), (SRC, args.expect_src)

from accelerate import Accelerator  # noqa: E402

from src.dataset import MSCOCOTestDataset  # noqa: E402
from src.model.mmae import MultiModalFusionMAE_CLIP_MultiLearner  # noqa: E402
from src.utils import setup_seed  # noqa: E402

eval_mod = importlib.import_module("src.hook.eval_fusionmmae")

REC = {"src": SRC, "args": vars(args), "error": None}
CAPTURE = {}

_real_encode = eval_mod.encode_data


def encode_wrapper(*a, **kw):
    out = _real_encode(*a, **kw)
    img, txt, t2i, i2t = out
    CAPTURE.update(img=img.cpu(), txt=txt.cpu(), t2i=t2i.cpu(), i2t=i2t.cpu())
    REC["num_images"] = int(img.shape[0])
    REC["num_texts"] = int(txt.shape[0])
    REC["t2i_map"] = t2i.tolist()
    return out


eval_mod.encode_data = encode_wrapper


def main():
    accelerator = None
    if args.mode == "accel":
        accelerator = Accelerator(cpu=(args.device == "cpu"))
        device = accelerator.device
        REC["num_processes"] = accelerator.num_processes
        REC["distributed_type"] = str(accelerator.distributed_type)
    else:
        device = torch.device("cuda" if args.device == "gpu" else "cpu")
        REC["num_processes"] = 1
    is_main = accelerator is None or accelerator.is_main_process

    setup_seed(0)
    model = MultiModalFusionMAE_CLIP_MultiLearner(
        image_size=224,
        patch_size=16,
        emb_dim=768,
        decoder_layer=4,
        decoder_head=8,
        mask_ratio=0.75,
        backbone_vision="openai/clip-vit-base-patch32",
        text_backbone="openai/clip-vit-base-patch32",
        proj_dim=256,
        fusion_method="concat",
    ).to(device)
    if args.model == "clip":
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        model.encode_image_tokens_cls = lambda images: clip.get_image_features(
            pixel_values=images
        ).pooler_output
        model.encode_text_tokens_cls = lambda ids: clip.get_text_features(
            input_ids=ids
        ).pooler_output

    ds = MSCOCOTestDataset(
        root="/data/SSD/coco/images/",
        split="test",
        image_size=224,
        tokenizer_name="openai/clip-vit-base-patch32",
        max_len=32,
    )
    if args.n > 0:  # --n 0 keeps the full split
        step = len(ds) // args.n
        ds = Subset(ds, list(range(0, step * args.n, step))[: args.n])
    loader = DataLoader(
        ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
    )

    try:
        if accelerator is not None:
            if args.unprepared:
                model_p = accelerator.prepare(model)
                loader_p = loader
            else:
                model_p, loader_p = accelerator.prepare(model, loader)
            REC["model_type"] = type(model_p).__name__
            REC["state_dict_first_key"] = next(iter(model_p.state_dict()))
            model_arg = model_p
            if args.variant in ("unwrap", "unwrap_cuda"):
                model_arg = accelerator.unwrap_model(model_p)
            if args.variant == "unwrap_cuda":
                torch.Tensor.cuda = lambda self, *a, **k: self
            metrics = eval_mod.evalrank(model_arg, loader_p, accelerator=accelerator)
        else:
            REC["model_type"] = type(model).__name__
            metrics = eval_mod.evalrank(model, loader)
        REC["metrics"] = {k: float(v) for k, v in metrics.items()}
    except Exception as e:
        REC["error"] = f"{type(e).__name__}: {e}"
        REC["traceback"] = traceback.format_exc()
        raise
    finally:
        if is_main:
            with open(args.out, "w") as f:
                json.dump(REC, f, indent=1)
            if CAPTURE:
                torch.save(CAPTURE, args.out.replace(".json", ".emb.pt"))
            print(f"[eval_check] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
