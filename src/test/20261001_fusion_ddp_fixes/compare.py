"""Summarise the run_all.sh outputs into the checks reported in the log.

Usage: python compare.py > summary.txt
"""
import ast
import json
import os
import re

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
O = os.environ.get("HARNESS_OUT") or os.path.join(_HERE, "out")
S = os.environ.get("HARNESS_DIR") or os.path.join(_HERE, "out", "work")
LOSS_KEYS = ["total", "mae", "mlm", "contrastive"]


def load(name):
    path = f"{O}/{name}.json"
    return json.load(open(path)) if os.path.exists(path) else None


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def step_losses(r, kind="train"):
    return {
        k: [s[f"{kind}/step_{k}_loss"] for s in r[f"{kind}_steps"]] for k in LOSS_KEYS
    }


def short(err, n=160):
    return (err or "")[:n].replace("\n", " ")


def section_a(det=False):
    sfx = "_det" if det else ""
    title = "F. Section A with torch.use_deterministic_algorithms(True)" if det else "A. Single GPU, baseline vs fixed (same seed, default GPU kernels)"
    print(f"\n## {title}\n")
    pairs = [
        (f"base_gpu_multi{sfx}_r1", f"base_gpu_multi{sfx}_r2"),
        (f"base_gpu_multi{sfx}_r1", f"fixed_gpu_multi{sfx}"),
        (f"base_gpu_plain{sfx}", f"fixed_gpu_plain{sfx}"),
    ]
    for a, b in pairs:
        ra, rb = load(a), load(b)
        if ra is None or rb is None:
            print(f"{a} vs {b}: missing")
            continue
        print(f"### {a} vs {b}")
        print(f"src: {ra['code']} | {rb['code']}")
        print(f"errors: {ra['error']!r} | {rb['error']!r}")
        la, lb = step_losses(ra), step_losses(rb)
        for k in LOSS_KEYS:
            print(
                f"  train/step_{k}_loss  n={len(la[k])}/{len(lb[k])}  bitwise identical={la[k] == lb[k]}  max|diff|={maxdiff(la[k], lb[k]):.3g}"
            )
            print(f"     {a}: {[round(x, 6) for x in la[k]]}")
            print(f"     {b}: {[round(x, 6) for x in lb[k]]}")
        va, vb = step_losses(ra, "val"), step_losses(rb, "val")
        print(f"  val step losses bitwise identical: {va == vb}  max|diff|={max(maxdiff(va[k], vb[k]) for k in LOSS_KEYS):.3g}")
        for ep in range(len(ra["val_retrieval"])):
            ma, mb = ra["val_retrieval"][ep], rb["val_retrieval"][ep]
            print(f"  epoch {ep + 1} val_retrieval identical: {ma == mb}")
            print(f"     {a}: {ma}")
            if ma != mb:
                print(f"     {b}: {mb}")
        ta, tb = ra["test_retrieval"], rb["test_retrieval"]
        print(f"  test_retrieval identical: {ta == tb}")
        for k in ["val_total_losses", "test_total_loss", "best_val_loss"]:
            print(f"  {k}: {ra['results'][k]} | {rb['results'][k]}")
        print(f"  best_epoch: {ra['results'].get('best_epoch')} | {rb['results'].get('best_epoch')}")
        print(f"  trainable/frozen params: {ra.get('trainable_params')}/{ra.get('frozen_params')} | {rb.get('trainable_params')}/{rb.get('frozen_params')}")
        print()


def restore_line(name):
    r = load(name)
    if r is None:
        return f"{name}: missing"
    calls = r["evalrank_calls"]
    val = [c["hash"] for c in calls if c["phase"] == "train"]
    test = [c["hash"] for c in calls if c["phase"] == "test"]
    best = r.get("results", {}).get("best_epoch")
    vloss = r.get("results", {}).get("val_total_losses")
    t = test[0] if test else None
    which = "epoch1" if t == val[0] else ("epoch2(last)" if t == val[-1] else "?")
    if best is None and "results" in r:
        which += " (no best epoch recorded)"
    return (
        f"{name}: val losses={vloss} best_epoch={best} | hash after ep1={val[0]} ep2={val[-1]} | "
        f"test retrieval hash={t} test-loss hash={r['test_loss_hash']} -> test used {which}"
    )


def section_b():
    print("\n## B. Best-weights restore (min_delta=1e9: epoch 1 best, epoch 2 last)\n")
    for n in ["base_gpu_multi_best1", "fixed_gpu_multi_best1", "base_gpu_plain_best1", "fixed_gpu_plain_best1", "fixed_cpu2_multi_best1", "base_gpu_multi_det_best1", "fixed_gpu_multi_det_best1", "fixed_gpu_multi_noimprove"]:
        print(restore_line(n))
    r1 = f"{O}/fixed_cpu2_multi_best1.rank1.json"
    if os.path.exists(r1):
        print("fixed_cpu2_multi_best1 rank1:", restore_line("fixed_cpu2_multi_best1.rank1"))
    for a, b in [("base_gpu_multi_best1", "fixed_gpu_multi_best1"), ("base_gpu_plain_best1", "fixed_gpu_plain_best1"), ("base_gpu_multi_det_best1", "fixed_gpu_multi_det_best1")]:
        ra, rb = load(a), load(b)
        if ra and rb:
            print(f"{a} vs {b}: train step losses identical={step_losses(ra) == step_losses(rb)}; epoch-1 val_retrieval identical={ra['val_retrieval'][0] == rb['val_retrieval'][0]}; test_retrieval identical={ra['test_retrieval'] == rb['test_retrieval']}")
            print(f"   test_retrieval base (last weights):  {ra['test_retrieval']}")
            print(f"   test_retrieval fixed (best weights): {rb['test_retrieval']}")


def section_c():
    print("\n## C. Two CPU processes (gloo) through the real hooks\n")
    for n in ["base_cpu2_multi", "base_cpu2_plain", "fixed_cpu2_multi", "fixed_cpu2_plain", "ablate_nofreeze_cpu2_multi", "ablate_noprep_test_cpu2_multi"]:
        r = load(n)
        if r is None:
            print(f"{n}: missing")
            continue
        enc = [(c["num_images"], c["num_texts"]) for c in r["encode_data_calls"]]
        print(f"{n}: src={r['code']} {r.get('distributed_type')} procs={r.get('num_processes')} train_steps={len(r['train_steps'])} error={short(r['error'])!r}")
        print(f"   evalrank (images, texts) per call [val ep1, val ep2, test]: {enc}  (expected [(41, 205), (41, 205), (43, 215)])")
        rk1 = f"{O}/{n}.rank1.json"
        if os.path.exists(rk1) and not r["error"]:
            r1 = json.load(open(rk1))
            h0 = [c["hash"] for c in r["evalrank_calls"]]
            h1 = [c["hash"] for c in r1["evalrank_calls"]]
            print(f"   trainable-param hashes rank0={h0} rank1={h1} equal={h0 == h1}")
            m0 = r["val_mean_across_processes"]
            m1 = r1["val_mean_across_processes"]
            if m0:
                print("   val total loss per epoch (local rank0, local rank1 -> global rank0, global rank1):")
                for i in range(0, len(m0), 4):
                    print(f"     {m0[i]['local']:.6f}, {m1[i]['local']:.6f} -> {m0[i]['global']:.6f}, {m1[i]['global']:.6f}")
        if r.get("results"):
            print(f"   test_retrieval: {r['test_retrieval']}")
    log = f"{O}/ablate_nofreeze_cpu2_multi.log"
    if os.path.exists(log):
        txt = open(log).read()
        m = re.search(r"Parameters which did not receive grad for rank 0: (.*)", txt)
        if m:
            names = m.group(1).split(", ")
            enc = [x for x in names if x.startswith(("vision_encoder.", "text_encoder."))]
            print(f"   ablate_nofreeze rank0: {len(names)} params without grad, {len(enc)} under vision_encoder./text_encoder., others={[x for x in names if x not in enc][:5]}")
            print(f"   e.g. {names[:2]} ... {names[-2:]}")
    ck = f"{S}/ckpt_fixed_cpu2/fusion_mmae_epoch_2.pth"
    if os.path.exists(ck):
        sd = torch.load(ck, map_location="cpu", weights_only=False)["fusion_model"]
        keys = list(sd)
        print(f"   checkpoint {ck.split('/')[-1]}: {len(keys)} keys, any 'module.' prefix={any(k.startswith('module.') for k in keys)}, first={keys[0]}")


def section_d():
    print("\n## D. evalrank on a fixed model\n")
    rows = [
        "eval_fixed_cpu1", "eval_fixed_cpu2", "eval_base_cpu1_unwrap_cuda",
        "eval_base_cpu2_as_is", "eval_base_cpu2_unwrap", "eval_base_cpu2_unwrap_cuda",
        "eval_fixed_cpu1_clip", "eval_fixed_cpu2_clip", "eval_base_cpu1_unwrap_cuda_clip", "eval_base_cpu2_unwrap_cuda_clip",
        "eval_noaccel_base_gpu", "eval_noaccel_fixed_gpu",
    ]
    for n in rows:
        r = load(n)
        if r is None:
            print(f"{n}: missing")
            continue
        print(f"{n}: src={r['src'].split('/')[-2]} procs={r.get('num_processes')} model={r.get('model_type')} sd_key0={r.get('state_dict_first_key')} images={r.get('num_images')} texts={r.get('num_texts')} error={short(r['error'])!r}")
        if r.get("metrics"):
            print(f"   {r['metrics']}")
    for a, b in [
        ("eval_fixed_cpu1", "eval_fixed_cpu2"),
        ("eval_fixed_cpu1", "eval_base_cpu1_unwrap_cuda"),
        ("eval_fixed_cpu1", "eval_base_cpu2_unwrap_cuda"),
        ("eval_fixed_cpu1_clip", "eval_fixed_cpu2_clip"),
        ("eval_fixed_cpu1_clip", "eval_base_cpu1_unwrap_cuda_clip"),
        ("eval_fixed_cpu1_clip", "eval_base_cpu2_unwrap_cuda_clip"),
        ("eval_noaccel_base_gpu", "eval_noaccel_fixed_gpu"),
    ]:
        ra, rb = load(a), load(b)
        if not (ra and rb and ra.get("metrics") and rb.get("metrics")):
            continue
        same = ra["metrics"] == rb["metrics"]
        diff = {k: (ra["metrics"][k], rb["metrics"][k]) for k in ra["metrics"] if ra["metrics"][k] != rb["metrics"][k]}
        line = f"{a} vs {b}: metrics identical={same}"
        if not same:
            line += f" differing={diff}"
        ea, eb = f"{O}/{a}.emb.pt", f"{O}/{b}.emb.pt"
        if os.path.exists(ea) and os.path.exists(eb):
            A, B = torch.load(ea), torch.load(eb)
            if A["img"].shape == B["img"].shape:
                line += (
                    f"; img emb max|diff|={(A['img'] - B['img']).abs().max().item():.3g}"
                    f" txt emb max|diff|={(A['txt'] - B['txt']).abs().max().item():.3g}"
                    f" bitwise={torch.equal(A['img'], B['img']) and torch.equal(A['txt'], B['txt'])}"
                    f" maps equal={torch.equal(A['t2i'], B['t2i']) and torch.equal(A['i2t'], B['i2t'])}"
                )
            else:
                line += f"; shapes differ img {tuple(A['img'].shape)} vs {tuple(B['img'].shape)}"
                line += f"; {b} t2i map tail={B['t2i'][-8:].tolist()}"
        print(line)


def section_e():
    print("\n## E. No-accelerator path at test_eval_fusionmmae.py scale (full test split, batch 256)\n")
    for n in ["testeval_script_base", "testeval_script_fixed"]:
        p = f"{O}/{n}.log"
        if not os.path.exists(p):
            print(f"{n}: missing")
            continue
        txt = open(p).read()
        src = re.search(r"src: (\S+)", txt)
        err = [l for l in txt.splitlines() if "Error" in l]
        print(f"{n}: src={src.group(1) if src else None} last error={err[-1] if err else None}")
    a, b = load("eval_noaccel_full_base_gpu"), load("eval_noaccel_full_fixed_gpu")
    if a and b:
        for n, r in [("eval_noaccel_full_base_gpu", a), ("eval_noaccel_full_fixed_gpu", b)]:
            print(f"{n}: src={r['src']} images={r['num_images']} texts={r['num_texts']} error={r['error']!r}")
            print(f"   {r['metrics']}")
        pa, pb = f"{O}/eval_noaccel_full_base_gpu.emb.pt", f"{O}/eval_noaccel_full_fixed_gpu.emb.pt"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"metrics identical={a['metrics'] == b['metrics']}; (embedding dumps deleted to save space; rerun section E)")
            return
        A, B = torch.load(pa), torch.load(pb)
        print(
            f"metrics identical={a['metrics'] == b['metrics']}; embeddings bitwise="
            f"{torch.equal(A['img'], B['img']) and torch.equal(A['txt'], B['txt'])}; "
            f"maps equal={torch.equal(A['t2i'], B['t2i']) and torch.equal(A['i2t'], B['i2t'])}"
        )


if __name__ == "__main__":
    print(open(f"{O}/exit_codes.txt").read() if os.path.exists(f"{O}/exit_codes.txt") else "")
    section_a()
    section_a(det=True)
    section_b()
    section_c()
    section_d()
    section_e()
