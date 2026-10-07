"""D4, D6 and D7 results from the per-run encode folders (spec sections 6 to 9): every readout and baseline with its
own temperature, agreement with the dense human votes, the human references, the four pre-registered bootstrap
tests with the decision, the view-sampling table and the partial-caption table. CPU only."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.special import softmax

from mmae.engine.hb import analysis, baselines, bootstrap, calibrate, data, metrics
from mmae.engine.hb.encode import D7_PATTERNS

ARMS = ("ml80", "ml80_mae", "parcap", "c")
LABELS = {"ml80": "ML-80", "ml80_mae": "ML-80+MAE", "parcap": "Par-cap", "c": "C"}
KILL_PATTERNS = ("prefix2", "prefix4", "prefix8", "random2", "random4", "random8")
MIN_SEEDS = 3


def arm_of(meta: dict, folder) -> str:
    if not meta.get("decoder"):
        return "c"
    source, ratio, mae = meta.get("mlm_image_source"), meta.get("text_ratio"), meta.get("mae_weight")
    if mae is not None:
        if source == "masked" and ratio == 0.8:
            return "ml80" if mae == 0 else "ml80_mae"
        if source == "clean" and ratio == 1.0:
            return "parcap"
    raise ValueError(f"{folder}: unknown arm (mlm_image_source={source!r}, text_ratio={ratio!r}, mae_weight={mae!r})")


def primary_name(arm: str, meta: dict) -> str:
    """Each arm's primary readout; C has no decoder, its prompt softmax is the H-b baseline as phrased."""
    return "prompt_full" if arm == "c" else analysis.primary_readout(meta)


def load_runs(root: str | Path) -> tuple[dict[str, dict[int, dict]], list[dict]]:
    runs, listing = {}, []
    for folder in sorted(Path(root).iterdir()):
        if not (folder / "meta.json").is_file():
            continue
        enc = analysis.load_encoded(folder)
        arm, seed = arm_of(enc["meta"], folder), enc["meta"].get("seed")
        if seed is None or seed in runs.get(arm, {}):
            raise ValueError(f"{folder}: missing or duplicate seed {seed!r} for arm {arm}")
        enc["_folder"] = folder.name
        check_d7(enc, arm, folder)
        runs.setdefault(arm, {})[int(seed)] = enc
        listing.append({"folder": folder.name, "arm": arm, "seed": int(seed)})
    if not runs:
        raise ValueError(f"no run folders (with meta.json) under {root}")
    encs = [e for by_seed in runs.values() for e in by_seed.values()]
    for e in encs[1:]:
        if e["meta"]["al28"] != encs[0]["meta"]["al28"] or not np.array_equal(e["al28_counts"], encs[0]["al28_counts"]):
            raise ValueError("runs disagree on the AL-28 paintings or counts")
        if (e["meta"]["val"] != encs[0]["meta"]["val"] or not np.array_equal(e["val_index"], encs[0]["val_index"])
                or not np.array_equal(e["val_labels"], encs[0]["val_labels"])):
            raise ValueError(f"{e['_folder']}: validation paintings, index or labels differ from {encs[0]['_folder']}")
    return runs, listing


def check_d7(enc: dict, arm: str, folder) -> None:
    """Every decoder run must carry the D7 arrays; masked-source arms were encoded on 4 views, Par-cap on 1."""
    if arm == "c":
        return
    if "d7_real" not in enc or "d7_null" not in enc:
        raise ValueError(f"{folder}: decoder run without D7 arrays (encoded with --skip-d7?)")
    want = 1 if arm == "parcap" else 4
    if enc["meta"].get("d7_views") != want or enc["d7_real"].shape[2] != want or enc["d7_null"].shape[2] != want:
        raise ValueError(f"{folder}: D7 must use {want} view(s) for {arm}, got meta d7_views={enc['meta'].get('d7_views')!r}, "
                         f"d7_real {enc['d7_real'].shape}")


def agreement(p: np.ndarray, h: np.ndarray, cuts: np.ndarray | None = None) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a constant model entropy gives a NaN Spearman (reported as null)
        out = {"jsd": float(metrics.js_distance(p, h).mean()), "entropy_spearman": metrics.entropy_spearman(p, h),
               "kl": float(metrics.kl(h, p).mean()), "tvd": float(metrics.tvd(p, h).mean()),
               "rank_cs": metrics.rank_cs(p, h)}
        if cuts is not None:
            third = np.digitize(metrics.entropy_bits(h), cuts)
            out["thirds"] = [{"n": int((third == t).sum()),
                              "jsd": float(metrics.js_distance(p[third == t], h[third == t]).mean()),
                              "entropy_spearman": metrics.entropy_spearman(p[third == t], h[third == t])}
                             if (third == t).sum() >= 3 else None for t in range(3)]
    return out


def primary_pair(p: np.ndarray, h: np.ndarray) -> dict:
    a = agreement(p, h)
    return {"jsd": a["jsd"], "entropy_spearman": a["entropy_spearman"]}


def readout_logits(enc: dict, prior: np.ndarray, probe: baselines.SoftProbe) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r = dict(analysis.decoder_readouts(enc))
    scale, prompts = float(enc["logit_scale"]), enc["prompts"]
    for tag, key in (("full", "emb_full"), ("views16", "emb_views")):
        r[f"prompt_{tag}"] = tuple(baselines.prompt_logits(enc[f"{s}_{key}"], prompts, scale) for s in ("al28", "val"))
        r[f"probe_{tag}"] = tuple(probe.logits(enc[f"{s}_{key}"]) for s in ("al28", "val"))
    r["prior"] = tuple(np.broadcast_to(np.log(prior), (len(enc[f"{s}_emb_full"]), 1, 9)) for s in ("al28", "val"))
    return r


def evaluate_run(enc: dict, probe, prior, target, counts, counts_od, cuts) -> tuple[dict, dict]:
    """(per-readout results, per-readout AL-28 distributions) for one (arm, seed)."""
    index, labels = enc["val_index"], enc["val_labels"]
    res, dist = {}, {}
    named = counts[:, :8].sum(1) > 0
    human8 = metrics.normalise(counts[named][:, :8])
    target_od = metrics.normalise(counts_od)
    for name, (al, val) in readout_logits(enc, prior, probe).items():
        t = calibrate.fit_temperature(val, index, labels)
        p = calibrate.mixture(al, t)
        p8 = p[named][:, :8]
        res[name] = {"T": t, "t_at_bound": bool(abs(np.log(t)) >= 2.99), "val_nll": calibrate.nll(calibrate.mixture(val, t), index, labels),
                     "metrics": agreement(p, target, cuts), "other_dropped": primary_pair(p, target_od),
                     "named8": {**primary_pair(p8 / p8.sum(1, keepdims=True), human8), "n": int(named.sum())}}
        dist[name] = p
    return res, dist


def human_references(names, counts, csv, min_votes, annotations_dir, cuts) -> dict:
    target = metrics.normalise(counts)
    votes = data.al28_votes(csv, min_votes)
    rng = np.random.default_rng(0)
    jsd, rho = [], []
    for _ in range(10):
        halves = ([], [])
        for n in names:
            v = votes[n][rng.permutation(len(votes[n]))]
            for half, part in zip(halves, (v[: len(v) // 2], v[len(v) // 2:])):
                half.append(np.bincount(part, minlength=9))
        a, b = (metrics.normalise(np.stack(h)) for h in halves)
        jsd.append(float(metrics.js_distance(a, b).mean()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho.append(metrics.entropy_spearman(a, b))
    out = {"split_half": {"jsd": float(np.mean(jsd)), "entropy_spearman": float(np.mean(rho)), "splits": 10, "seed": 0,
                          "n": len(names)}, "english": None}
    if annotations_dir is not None:
        english = data.english_counts(list(names), annotations_dir)
        has = english.sum(1) > 0
        eng = agreement(metrics.normalise(english[has]), target[has])
        eng.pop("kl")  # meaningless: the 5-vote histograms hold zeros
        out["english"] = {**eng, "n": int(has.sum()),
                          "n_dropped_without_english": int((~has).sum())}
    return out


def strongest_probes(arms: dict, seeds: list[int]) -> dict[int, dict]:
    """Per seed, the arm whose probe has the lowest validation NLL (at T = 1, as fitted); ties go to ARMS order."""
    return {s: min(((arm, arms[arm][s]["probe"]["val_nll"]) for arm in ARMS if arm in arms), key=lambda x: x[1])
            for s in seeds}


def pool_note(present) -> str | None:
    """Names the strongest-probe pool when an arm is absent (the pool, hence the comparator, is then smaller)."""
    absent = [LABELS[a] for a in ARMS if a not in present]
    if not absent:
        return None
    return f"missing arm(s): {', '.join(absent)}; probe pool: {', '.join(LABELS[a] for a in ARMS if a in present)}"


def _provisional(seed_sets: dict[str, list[int]], pool: str | None = None) -> str | None:
    if all(len(v) >= MIN_SEEDS for v in seed_sets.values()) and not pool:
        return None
    parts = [f"seeds: " + "; ".join(f"{k} {v}" for k, v in seed_sets.items())] + ([pool] if pool else [])
    return "provisional (" + "; ".join(parts) + ")"


def _test_entry(r: dict, extra: dict) -> dict:
    return {"diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"], **extra}


def pre_registered_tests(dist, seeds, best, target, B, pool=None) -> tuple[dict, dict, dict]:
    ml_seeds, pc_seeds = seeds["ml80"], seeds["parcap"]
    a = [dist["ml80"][s][primary_name("ml80", seeds["meta"]["ml80"])] for s in ml_seeds]
    comparators = {"parcap": ([dist["parcap"][s]["full"] for s in pc_seeds], {"ml80": ml_seeds, "parcap": pc_seeds}),
                   "probe": ([dist[best[s][0]][s]["probe_full"] for s in seeds["complete"]],
                             {"ml80": ml_seeds, "probe (complete seeds)": seeds["complete"]})}
    raw = {(m, c): bootstrap.paired_bootstrap(a, b, target, m, B=B, seed=0)
           for m in bootstrap.METRICS for c, (b, _) in comparators.items()}
    decision = bootstrap.decide(raw)
    tests, notes = {}, []
    for (m, c), r in raw.items():
        note = _provisional(comparators[c][1], pool)
        notes.append(note)
        tests[f"{m}/{c}"] = _test_entry(r, {"holm_p": decision["holm"][f"{m}/{c}"], "favours": decision["favours"][f"{m}/{c}"],
                                            "provisional": note})
    pending = [n for n in notes if n]
    decision = {**decision, "provisional": pending[0] if pending else None,
                "kill": not decision["support"],
                "rule": "support if, for at least one metric, ML-80 beats both Par-cap and the strongest probe with "
                        "Holm-adjusted p < 0.05 (Holm over the four tests)"}
    secondary = {}
    if "ml80_mae" in dist:
        mae_seeds = seeds["ml80_mae"]
        b = [dist["ml80_mae"][s][primary_name("ml80_mae", seeds["meta"]["ml80_mae"])] for s in mae_seeds]
        for m in bootstrap.METRICS:  # ML-80+MAE (a) minus ML-80 (b); no Holm
            r = bootstrap.paired_bootstrap(b, a, target, m, B=B, seed=0)
            secondary[f"{m}/ml80_mae_vs_ml80"] = _test_entry(r, {"provisional": _provisional(
                {"ml80_mae": mae_seeds, "ml80": ml_seeds}, pool)})
    return tests, secondary, decision


def between_view_mi(enc: dict, temperature: float) -> np.ndarray:
    """Per painting H(mean over views of p_v) - mean_v H(p_v), bits; p_v averages the softmax over lengths."""
    z = enc["al28_dec_views"].astype(np.float64) / temperature  # (N, V, L, 9)
    p = softmax(z, axis=-1).mean(axis=2)
    n, v, _ = p.shape
    return metrics.entropy_bits(p.mean(1)) - metrics.entropy_bits(p.reshape(-1, 9)).reshape(n, v).mean(1)


def boot_spearman(mis: list[np.ndarray], he: np.ndarray, B: int, seed: int = 0) -> np.ndarray:
    """Mean over seeds of Spearman(mi, he); each replicate resamples the paintings (shared by the seeds) and the seeds."""
    rng, n, S = np.random.default_rng(seed), len(he), len(mis)
    draws = np.empty(B)
    for r in range(B):
        idx, drawn = rng.integers(0, n, n), rng.integers(0, S, S)
        rho = [bootstrap._spearman_rows(m[idx], he[idx]) for m in mis]
        draws[r] = np.mean([rho[i] for i in drawn])
    return draws


def d6_tables(runs, arms, dist, best, seeds, target, B, pool=None) -> dict:
    he = metrics.entropy_bits(target)
    out = {"K": {"views1": 1, "views4": 4, "views16": 16}, "arms": {}, "controls": {}}

    def pair(arm, s, name):
        return {k: arms[arm][s]["readouts"][name]["metrics"][k] for k in ("jsd", "entropy_spearman")}

    def mean_pair(items):
        return {k: float(np.mean([i[k] for i in items])) for k in ("jsd", "entropy_spearman")}

    for arm in ("ml80", "ml80_mae"):
        if arm not in arms:
            continue
        ss = seeds[arm]
        res = {name: {"per_seed": {str(s): pair(arm, s, name) for s in ss},
                      "mean": mean_pair([pair(arm, s, name) for s in ss])} for name in ("views1", "views4", "views16", "full")}
        k16, k1 = res["views16"]["mean"], res["views1"]["mean"]
        res["k16_beats_k1"] = {"jsd": k16["jsd"] < k1["jsd"], "entropy_spearman": k16["entropy_spearman"] > k1["entropy_spearman"]}
        mis = [between_view_mi(runs[arm][s], arms[arm][s]["readouts"]["views16"]["T"]) for s in ss]
        draws = boot_spearman(mis, he, B)
        per_seed = [bootstrap._spearman_rows(m, he) for m in mis]
        lo, hi = (float(x) for x in np.nanquantile(draws, [0.025, 0.975]))
        res["mi"] = {"spearman_per_seed": {str(s): float(x) for s, x in zip(ss, per_seed)}, "spearman_mean": float(np.mean(per_seed)),
                     "ci_low": lo, "ci_high": hi, "interval_includes_zero": bool(lo <= 0 <= hi),
                     "nan_draws": int(np.isnan(draws).sum())}
        out["arms"][arm] = res
    controls = {"parcap_views16": [pair("parcap", s, "views16") for s in seeds["parcap"]],
                "c_prompt_views16": [pair("c", s, "prompt_views16") for s in seeds.get("c", [])],
                "strongest_probe_views16": [pair(best[s][0], s, "probe_views16") for s in seeds["complete"]]}
    out["controls"] = {k: mean_pair(v) for k, v in controls.items() if v}
    out["missing_controls"] = [k for k, v in controls.items() if not v]
    out["probe_pool_note"] = pool
    return out


def _boot_mean(d: np.ndarray, B: int, seed: int = 0, chunk: int = 100) -> tuple[float, float]:
    """95% interval of the mean of d (seeds, captions): each replicate resamples the captions (shared by the seeds)
    and the seeds, then averages over the drawn seeds."""
    rng, means = np.random.default_rng(seed), []
    S, n = d.shape
    for start in range(0, B, chunk):
        c = min(chunk, B - start)
        idx, drawn = rng.integers(0, n, (c, n)), rng.integers(0, S, (c, S))
        per_seed = d[:, idx].mean(2)  # (S, c)
        means.append(np.take_along_axis(per_seed, drawn.T, 0).mean(0))
    lo, hi = np.quantile(np.concatenate(means), [0.025, 0.975])
    return float(lo), float(hi)


def d7_tables(runs, arms, seeds, B) -> dict:
    out = {"patterns": list(D7_PATTERNS), "arms": {}}
    for arm in ("ml80", "ml80_mae", "parcap"):
        if arm not in runs or "d7_real" not in runs[arm][seeds[arm][0]]:
            continue
        ss = seeds[arm]
        name = "views4" if arm != "parcap" else "full"
        valid, label = runs[arm][ss[0]]["d7_valid"].astype(bool), runs[arm][ss[0]]["d7_label"]
        per_seed = {"real": [], "null": []}
        for s in ss:
            e = runs[arm][s]
            if not np.array_equal(e["d7_valid"].astype(bool), valid) or not np.array_equal(e["d7_label"], label):
                raise ValueError(f"{arm}: d7_valid or d7_label differ between seeds")
            t = arms[arm][s]["readouts"][name]["T"]
            for kind in per_seed:
                p = softmax(e[f"d7_{kind}"].astype(np.float64) / t, axis=-1).mean(axis=2)  # (n, patterns, 9)
                per_seed[kind].append(p)
        rows = {}
        for k, pattern in enumerate(D7_PATTERNS):
            v, y = valid[:, k], label[valid[:, k]]
            row = {"n": int(v.sum())}
            ll, acc = {}, {}
            for kind in per_seed:
                p = np.stack([x[v, k] for x in per_seed[kind]])  # (seeds, n, 9)
                ll[kind] = -np.log(np.clip(np.take_along_axis(p, y[None, :, None].repeat(len(ss), 0), 2)[..., 0], 1e-12, None))
                acc[kind] = (p.argmax(-1) == y[None]).astype(float)
                row[kind] = {"log_loss": float(ll[kind].mean()) if len(y) else None,
                             "accuracy": float(acc[kind].mean()) if len(y) else None}
            if len(y):
                d = ll["real"] - ll["null"]  # (seeds, captions)
                lo, hi = _boot_mean(d, B)
                row["real_minus_null"] = {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi}
                if arm == "ml80" and pattern in KILL_PATTERNS:
                    row["no_image_benefit"] = bool(hi >= 0)
            rows[pattern] = row
        out["arms"][arm] = {"readout": name, "seeds": ss, "patterns": rows}
    return out


def build(encoded_root, annotations_dir=None, al28_csv=data.AL28_CSV, B: int = 10000, min_votes: int = 20) -> dict:
    runs, listing = load_runs(encoded_root)
    missing = [a for a in ("ml80", "parcap") if a not in runs]
    if missing:
        raise ValueError(f"the pre-registered tests need the arms {missing}")
    enc0 = next(iter(runs["ml80"].values()))
    names, counts = enc0["meta"]["al28"], enc0["al28_counts"].astype(np.int64)
    ref_p, ref_c = data.al28_targets(al28_csv, min_votes)
    if ref_p.names != list(names) or not np.array_equal(ref_c, counts):
        raise ValueError("the encode files' AL-28 paintings or counts differ from the AL-28 CSV")
    counts_od = data.al28_targets(al28_csv, min_votes, drop_other=True)[1]
    target = metrics.normalise(counts)
    cuts = np.quantile(metrics.entropy_bits(target), [1 / 3, 2 / 3])

    arms, dist, probes = {}, {}, {}
    for arm in ARMS:
        for seed, enc in sorted(runs.get(arm, {}).items()):
            probe = baselines.fit_soft_probe(enc["train_emb"], enc["train_counts"], enc["val_emb_full"],
                                             enc["val_index"], enc["val_labels"])
            res, d = evaluate_run(enc, probe, analysis.prior_from(enc, annotations_dir), target, counts, counts_od, cuts)
            arms.setdefault(arm, {})[seed] = {"readouts": res, "primary": primary_name(arm, enc["meta"]),
                                              "probe": {"weight_decay": probe.weight_decay, "val_nll": probe.val_nll}}
            dist.setdefault(arm, {})[seed] = d
    present = [a for a in ARMS if a in runs]
    seeds = {a: sorted(runs[a]) for a in present}
    seeds["complete"] = sorted(set.intersection(*(set(seeds[a]) for a in present)))
    if not seeds["complete"]:
        raise ValueError("no seed is present in every arm, so no strongest probe can be chosen")
    seeds["meta"] = {a: next(iter(runs[a].values()))["meta"] for a in present}
    best = strongest_probes(arms, seeds["complete"])

    tests, secondary, decision = pre_registered_tests(dist, seeds, best, target, B, pool_note(present))
    return {
        "runs": listing,
        "arms": {a: {str(s): v for s, v in by.items()} for a, by in arms.items()},
        "probes": {"per_arm": {a: {str(s): v["probe"] for s, v in by.items()} for a, by in arms.items()},
                   "strongest": {str(s): {"arm": a, "val_nll": v} for s, (a, v) in best.items()}},
        "references": human_references(names, counts, al28_csv, min_votes, annotations_dir, cuts),
        "tests": {**tests, "secondary": secondary}, "decision": decision,
        "d6": d6_tables(runs, arms, dist, best, seeds, target, B, pool_note(present)),
        "d7": d7_tables(runs, arms, seeds, B),
        "settings": {"B": B, "min_votes": min_votes, "seeds": {a: seeds[a] for a in present},
                     "complete_seeds": seeds["complete"], "entropy_third_cuts_bits": cuts.tolist()},
    }
