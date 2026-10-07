import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mmae.engine.hb import coco_probes as cp


def test_ap_at_r_known_values():
    order = np.array([[3, 1, 2, 0], [0, 1, 2, 3]])
    positives = [np.array([3, 2]), np.array([2, 3])]
    R = np.array([2, 2])
    np.testing.assert_allclose(cp.ap_at_r(order, positives, R), [0.5, 0.0])


def test_rerank_only_touches_the_top_k():
    order = np.array([[5, 4, 3, 2, 1, 0]])
    out = cp.rerank(order, np.array([[0.1, 0.9, 0.5]]))
    assert out.tolist() == [[4, 3, 5, 2, 1, 0]]


def test_topk_is_a_stable_cosine_sort():
    image = torch.tensor([[1.0, 0.0]])
    captions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
    assert cp.topk_captions(image, captions, np.array([0]), 3).tolist() == [[0, 2, 3]]


@pytest.mark.slow
def test_per_query_ap_reproduces_the_package():
    """Mean AP@R over the i2t queries equals the package's eccv/i2t_map_at_r (coco_test_metrics) on the zero-shot embeddings."""
    from eccv_caption import Metrics

    from mmae.data.coco import retrieval_items
    from mmae.engine.eccv import CAPTIONS_FILE, coco_test_metrics, map_coco_ids

    ann = Path("/data/SSD/coco/annotations")
    emb = torch.load(Path("res/coco/diagnostics/stage2/embeddings/zeroshot.pt"))
    q = cp.eccv_i2t(ann)
    image = torch.nn.functional.normalize(emb["image"].float(), dim=-1)
    cap = torch.nn.functional.normalize(emb["caption"].float(), dim=-1)
    order = cp.topk_captions(image, cap.reshape(-1, cap.shape[-1]), q.query_image, int(q.R.max()))
    ours = 100 * cp.ap_at_r(order, q.positives, q.R).mean()
    image_ids, caption_ids = map_coco_ids(retrieval_items(ann, "test"), ann / CAPTIONS_FILE)
    reported = coco_test_metrics(image, cap, image_ids, caption_ids, Metrics())["eccv/i2t_map_at_r"]
    assert abs(ours - reported) < 0.02, (ours, reported)


# ---- D1: decoder caption scores, tuning, script ----
from helpers import make_batch  # noqa: E402
from test_model import tiny_model  # noqa: E402


def _tok(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask")}


@pytest.mark.parametrize("kind", ["parallel", "ratio"])
@pytest.mark.parametrize("name", ["fusion_multilearner", "fusion_none"])
def test_caption_scores_are_finite_per_pair(name, kind, tokenizer):
    model = tiny_model(name).eval()
    batch = make_batch(tokenizer)
    out = cp.caption_scores(model, batch["pixel_values"], _tok(batch), kind, 3, 0)
    assert out.shape == (4,) and torch.isfinite(out).all() and (out <= 0).all()


def test_parallel_score_of_fusion_none_ignores_the_image(tokenizer):
    model = tiny_model("fusion_none").eval()
    batch = make_batch(tokenizer, batch_size=2, captions=["a dog on the beach"] * 2)
    other = torch.randn_like(batch["pixel_values"][:1])
    images = torch.cat([batch["pixel_values"][:1], other])
    s = cp.caption_scores(model, images, _tok(batch), "parallel", 8, 0)
    torch.testing.assert_close(s[0], s[1], rtol=0, atol=1e-5)
    # and a masked-source model does read the image
    ml = tiny_model("fusion_multilearner").eval()
    s2 = cp.caption_scores(ml, images, _tok(batch), "parallel", 2, 0)
    assert abs(s2[0] - s2[1]) > 1e-6


def test_ratio_score_depends_on_the_seed_and_parallel_scores_every_token(tokenizer):
    model = tiny_model("fusion_multilearner").eval()
    batch = make_batch(tokenizer)
    a = cp.caption_scores(model, batch["pixel_values"], _tok(batch), "ratio", 2, 0)
    b = cp.caption_scores(model, batch["pixel_values"], _tok(batch), "ratio", 2, 100)
    assert not torch.allclose(a, b)
    again = cp.caption_scores(model, batch["pixel_values"], _tok(batch), "ratio", 2, 0)
    torch.testing.assert_close(a, again)


def test_parallel_score_is_the_mean_log_prob_of_real_tokens(tokenizer):
    model = tiny_model("fusion_none").eval()
    batch = make_batch(tokenizer, batch_size=2, captions=["a dog on the beach", "two men on big horses"])
    ids, am = batch["input_ids"], batch["attention_mask"]
    n = am.sum(1)
    real = am.bool().clone()
    real[:, 0] = False
    real[torch.arange(2), n - 1] = False
    with torch.no_grad():
        logits, _ = model.decode_text(batch["pixel_values"], ids, am, real)
    lp = torch.log_softmax(logits.float(), -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    expected = (lp * real).sum(1) / real.sum(1)
    out = cp.caption_scores(model, batch["pixel_values"], _tok(batch), "parallel", 1, 0)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-5)


def test_null_images_are_zeros():
    z = cp.null_images(3)
    assert z.shape == (3, 3, 224, 224) and z.abs().sum() == 0


def test_tune_prefers_no_correction_when_the_decoder_is_noise_and_dual_is_perfect():
    rng = np.random.default_rng(0)
    Q, k = 60, 50
    order = np.tile(np.arange(k), (Q, 1))
    positives = [np.arange(5) for _ in range(Q)]
    dual = np.zeros((Q, k))
    dual[:, :5] = 10.0
    dual += rng.normal(scale=0.01, size=(Q, k))
    out = cp.tune(dual, rng.normal(size=(Q, k)), rng.normal(size=(Q, k)), order, positives, 5)
    assert out["alpha_b"] == 0 and out["beta"] == 0


def test_tune_finds_the_prior_when_the_decoder_score_is_prior_plus_signal():
    rng = np.random.default_rng(1)
    Q, k = 80, 50
    order = np.tile(np.arange(k), (Q, 1))
    positives = [np.arange(5) for _ in range(Q)]
    prior = rng.normal(scale=3.0, size=(Q, k))
    signal = np.zeros((Q, k))
    signal[:, :5] = 2.0
    out = cp.tune(rng.normal(size=(Q, k)), signal + prior, prior, order, positives, 5)
    assert out["alpha_a"] == 1.0


def test_r_precision_and_thirds():
    order = np.array([[3, 1, 2, 0], [0, 1, 2, 3]])
    positives = [np.array([3, 2]), np.array([2, 3])]
    np.testing.assert_allclose(cp.r_precision(order, positives, np.array([2, 2])), [0.5, 0.0])
    groups = cp.r_thirds(np.array([10, 15, 16, 20, 21, 30]))
    assert [g.tolist() for g in groups.values()] == [[0, 1], [2, 3], [4, 5]]


def test_evaluate_orders_reranks_only_the_top_k():
    # 1 query, 6 captions, positives {0, 1}; dual order puts them at ranks 2 and 3 within top k=4
    order = np.array([[5, 4, 0, 1, 3, 2]])
    q = cp.EccvI2T(query_image=np.array([0]), positives=[np.array([0, 1])], R=np.array([2]))
    base = cp.d1_metrics(order, q)
    better = cp.d1_metrics(cp.rerank(order, np.array([[0.0, 0.0, 2.0, 1.0]])), q)
    assert better["map_at_r"] > base["map_at_r"] and better["r_precision"] == 100.0
    assert set(base["by_third"]) == {"R<=15", "R16-20", "R>=21"}


def _load_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("hb_coco_d1", Path(__file__).resolve().parents[1] / "scripts/hb_coco_d1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_run(tmp_path, fake_coco, name):
    from omegaconf import OmegaConf

    from helpers import compose_cfg
    from test_model import TINY_CLIP

    images_dir, annotations_dir = fake_coco
    cfg = compose_cfg(f"model={name}", f"model.backbone.pretrained={TINY_CLIP}", f"data.images_dir={images_dir}",
                      f"data.annotations_dir={annotations_dir}")
    # tiny random CLIP: the processor is B/32's (processor_name), config as the training run wrote it
    run = tmp_path / f"run_{name}"
    (run / "checkpoints").mkdir(parents=True)
    OmegaConf.save(cfg, run / "config.yaml")
    torch.manual_seed(0)
    from mmae.models import MultiMAE

    model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
    torch.save({"model": model.state_dict()}, run / "checkpoints" / "best.pt")
    (run / "run.json").write_text('{"status": "completed"}')
    return run


def _args(**kw):
    import argparse

    base = dict(images_dir=None, annotations_dir=None, k=50, samples=2, val_images=6, batch_size=60, workers=0,
                limit_queries=None, device="cpu")
    return argparse.Namespace(**{**base, **kw})


def test_d1_script_end_to_end_on_the_fake_coco(tmp_path, fake_coco, monkeypatch):
    script = _load_script()
    run = _make_run(tmp_path, fake_coco, "fusion_multilearner")
    q = cp.EccvI2T(query_image=np.arange(4), positives=[np.arange(5 * i, 5 * i + 5) for i in range(4)],
                   R=np.full(4, 5))
    monkeypatch.setattr(cp, "eccv_i2t", lambda ann: q)
    result, arrays = script.run_one(run, _args(k=10), torch.device("cpu"))
    assert result["meta"]["queries"] == 4 and result["meta"]["k"] == 10
    assert arrays["cand"].shape == (4, 10) and arrays["dec_ratio"].shape == (4, 10)
    for kind in ("parallel", "ratio"):
        assert set(result[kind]["grid_a"]) == {str(a) for a in cp.ALPHAS}
        assert len(result[kind]["grid_b"]) == len(cp.ALPHAS) * len(cp.BETAS)
        assert result[kind]["grid_a"]["0.0"]["by_third"]["R<=15"]["n"] == 4
    for kind in ("parallel", "ratio"):
        for key in ("rerank_a", "rerank_b", "raw_decoder"):
            assert 0 <= result[kind][key]["map_at_r"] <= 100
        assert result[kind]["tuned"]["alpha_a"] in cp.ALPHAS
    assert 0 <= result["dual"]["map_at_r"] <= 100
    # alpha = 0, beta = 0 reproduces the dual order exactly
    pass_through = cp.score_b(np.arange(30.0)[None, ::-1], np.random.rand(1, 30), np.random.rand(1, 30), 0.0, 0.0)
    assert np.argsort(-pass_through[0]).tolist() == list(range(30))
    one, one_arrays = script.run_one(run, _args(k=10, limit_queries=2), torch.device("cpu"))
    assert one_arrays["cand"].shape == (2, 10)
    assert one["meta"]["queries"] == 2 and one["meta"]["queries_total"] == 4 and one["dual"]["n"] == 2


def test_d1_script_skips_checkpoints_without_a_decoder(tmp_path, fake_coco):
    script = _load_script()
    run = _make_run(tmp_path, fake_coco, "contrastive")
    assert script.run_one(run, _args(), torch.device("cpu")) is None


def _fake_val(fake_coco):
    from mmae.data import Collator
    from mmae.data.coco import CocoRetrieval
    from mmae.data.transforms import build_image_transform
    images_dir, annotations_dir = fake_coco
    processor = "openai/clip-vit-base-patch32"
    dataset = CocoRetrieval(images_dir, annotations_dir, "val", build_image_transform(processor), 6)
    collator = Collator(processor, 32)
    tok = {k: v for k, v in collator.tokenize([c for _, caps in dataset.items for c in caps]).items()}
    cand = np.array([[(7 * i + 3 * j) % 30 for j in range(6)] for i in range(6)])
    return script_mod(), dataset, tok, cand


def script_mod():
    return _load_script()


def _score(model, script, dataset, tok, cand, batch_size, samples=3):
    d = script.draws(model, tok, len(dataset), samples)
    dec = script.score_pairs(model, dataset, np.arange(len(dataset)), cand, tok, script.KINDS, samples, batch_size, 0,
                             torch.device("cpu"), d)
    null = script.score_null(model, cand, tok, script.KINDS, samples, batch_size, torch.device("cpu"), d)
    return dec, null


@pytest.mark.parametrize("name", ["fusion_multilearner", "fusion_none", "fusion_concat"])
def test_scores_do_not_depend_on_batch_size(name, fake_coco):
    script, dataset, tok, cand = _fake_val(fake_coco)
    model = tiny_model(name).eval()
    dec_a, null_a = _score(model, script, dataset, tok, cand, batch_size=60)
    dec_b, null_b = _score(model, script, dataset, tok, cand, batch_size=12)  # 2 queries per chunk, null chunks of 12
    for kind in script.KINDS:
        np.testing.assert_allclose(dec_a[kind], dec_b[kind], atol=1e-5)
        np.testing.assert_allclose(null_a[kind], null_b[kind], atol=1e-5)


def test_pair_score_is_unchanged_when_the_chunk_order_changes(fake_coco):
    script, dataset, tok, cand = _fake_val(fake_coco)
    model = tiny_model("fusion_multilearner").eval()
    dec, _ = _score(model, script, dataset, tok, cand, batch_size=60)
    d = script.draws(model, tok, len(dataset), 3)
    perm = np.array([4, 2, 5, 0, 3, 1])
    rev = script.score_pairs(model, dataset, perm, cand[perm], tok, ("ratio",), 3, 60, 0, torch.device("cpu"), d)
    np.testing.assert_allclose(rev["ratio"], dec["ratio"][perm], atol=1e-5)


def test_fusion_none_ratio_pmi_at_alpha_one_is_zero(fake_coco):
    script, dataset, tok, cand = _fake_val(fake_coco)
    model = tiny_model("fusion_none").eval()
    dec, null = _score(model, script, dataset, tok, cand, batch_size=60)
    for kind in script.KINDS:
        np.testing.assert_allclose(dec[kind] - null[kind], 0.0, atol=1e-5)


def test_image_and_null_scores_use_the_same_token_masks_per_caption(fake_coco, monkeypatch):
    script, dataset, tok, cand = _fake_val(fake_coco)
    model = tiny_model("fusion_multilearner").eval()
    seen: dict[tuple, list] = {}
    original = model.decode_text

    def spy(pixel_values, input_ids, attention_mask, token_mask, ids_keep=None):
        zero = bool(pixel_values.abs().sum() == 0)
        for ids, m in zip(input_ids, token_mask):
            seen.setdefault(tuple(ids.tolist()), []).append((zero, m.clone()))
        return original(pixel_values, input_ids, attention_mask, token_mask, ids_keep=ids_keep)

    monkeypatch.setattr(model, "decode_text", spy)
    _score(model, script, dataset, tok, cand, batch_size=24)
    attention = tok["attention_mask"]
    checked = 0
    for ids, calls in seen.items():
        # parallel calls hide every real token; the remaining (ratio) masks must agree between image and null
        partial = [(z, m) for z, m in calls if int(m.sum()) < int(attention[tok["input_ids"].eq(torch.tensor(ids)).all(1)][0].sum()) - 2]
        img = {tuple(m.tolist()) for z, m in partial if not z}
        nul = {tuple(m.tolist()) for z, m in partial if z}
        if img and nul:
            assert img <= nul
            checked += 1
    assert checked


# ---- D2: blend probe helpers ----
def test_select_pairs_are_distinct_low_similarity_and_deterministic():
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(60, 5, 8, generator=g) + 3 * torch.randn(60, 1, 8, generator=g)
    pairs = cp.select_pairs(emb, n_pairs=12, n_random=2000)
    assert pairs.shape == (12, 2) and (pairs[:, 0] != pairs[:, 1]).all()
    assert len({tuple(sorted(p)) for p in pairs.tolist()}) == 12
    again = cp.select_pairs(emb, n_pairs=12, n_random=2000)
    np.testing.assert_array_equal(pairs, again)
    m = torch.nn.functional.normalize(emb, dim=-1).mean(1)
    sims = (m[pairs[:, 0]] * m[pairs[:, 1]]).sum(-1)
    rng = np.random.RandomState(1)
    a = rng.randint(0, 60, 2000)
    b = rng.randint(0, 59, 2000)
    b = b + (b >= a)
    thr = np.percentile((m[a] * m[b]).sum(-1).numpy(), 10)
    assert (sims.numpy() < thr).all()


def test_blend_with_lambda_one_is_image_a_exactly():
    a, b = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    assert torch.equal(cp.blend(a, b, 1.0), a) and torch.equal(cp.blend(a, b, 0.0), b)
    torch.testing.assert_close(cp.blend(a, b, 0.25), 0.25 * a + 0.75 * b)


def test_patch_mix_takes_m_patches_from_a_and_the_rest_from_b():
    a, b = torch.full((3, 224, 224), 1.0), torch.full((3, 224, 224), 2.0)
    comp, ids = cp.patch_mix(a, b, 5, torch.Generator().manual_seed(3))
    assert ids.shape == (13,) and len(set(ids.tolist())) == 13
    grid = comp[0].unfold(0, 32, 32).unfold(1, 32, 32).reshape(49, 32, 32)  # patch p = row * 7 + col
    kept = grid[ids]
    assert (kept[:5] == 1.0).all() and (kept[5:] == 2.0).all()
    rest = np.setdiff1d(np.arange(49), ids.numpy())
    assert (grid[rest] == 0).all()
    _, again = cp.patch_mix(a, b, 9, torch.Generator().manual_seed(3))
    assert torch.equal(ids, again)  # same positions for every m


def test_both_covered_and_balance_on_hand_made_orders():
    a_rows = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    b_rows = np.array([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])
    order = np.array([[0, 5, 11], [0, 1, 2]])
    assert cp.both_covered(order, a_rows, b_rows) == 0.5
    assert cp.both_covered(order[:, :1], a_rows, b_rows) == 0.0
    sa = np.array([[9, 8, 7, 0, 0], [1, 0, 0, 0, 0]], dtype=float)
    sb = np.array([[1, 1, 1, 1, 1], [5, 6, 7, 8, 9]], dtype=float)
    np.testing.assert_allclose(cp.balance(sa, sb), [0.6, 0.0])


def test_selection_index_drops_ties_and_is_one_at_the_source():
    s_a = np.array([-1.0, -2.0, -3.0])
    s_b = np.array([-3.0, -2.0, -1.0])
    idx, dropped = cp.selection_index(np.array([-1.0, -2.0, -2.0]), s_a, s_b)
    assert dropped == 1 and idx.tolist() == [1.0, 0.5]
    idx, dropped = cp.selection_index(s_a, s_a, s_b)
    assert dropped == 1 and (idx == 1.0).all()


# ---- D2: script ----
def _load_d2():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/hb_coco_d2.py"
    spec = importlib.util.spec_from_file_location("hb_coco_d2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _d2_shared(mod, fake_coco, limit=None):
    from mmae.data import Collator
    from mmae.data.coco import CocoRetrieval
    from mmae.data.transforms import build_image_transform
    images_dir, annotations_dir = fake_coco
    processor = "openai/clip-vit-base-patch32"
    dataset = CocoRetrieval(images_dir, annotations_dir, "test", build_image_transform(processor))
    pairs = np.array([[0, 1], [2, 3], [4, 5], [1, 4], [0, 5]])
    return mod.build_shared(dataset, Collator(processor, 32), pairs, limit)


def _d2_args(**kw):
    import argparse
    return argparse.Namespace(**{**dict(batch_size=100), **kw})


D1_JSON = {"parallel": {"tuned": {"alpha_a": 0.5}}}


def test_d2_decoder_model_gives_every_readout(fake_coco):
    mod = _load_d2()
    sh = _d2_shared(mod, fake_coco)
    model = tiny_model("fusion_multilearner").eval()
    result, arrays = mod.run_model(model, "m", sh, _d2_args(), torch.device("cpu"), D1_JSON)
    keys = {mod.cond_key(c) for c in mod.CONDITIONS}
    assert set(result["dual"]) == keys == set(result["decoder"]["balance_parallel"]) == set(result["decoder"]["both_covered_rerank"])
    assert set(result["dual"]["blend_0.5"]["both_covered"]) == {"5", "10", "20"}
    assert set(result["decoder"]["selection"]) == {"0", "1", "2", "3"}
    for j, entry in result["decoder"]["selection"].items():
        assert entry["cA"]["n"] + entry["cA"]["dropped"] == 5 and entry["pooled"]["n"] + entry["pooled"]["dropped"] == 10
    assert arrays["dual_balance_mix_3"].shape == (5,) and 0 <= result["dual"]["mix_3"]["balance"] <= 1
    assert result["meta"]["alpha_a_parallel"] == 0.5
    # without a D1 file the re-ranked rows are skipped, not invented
    no_d1, _ = mod.run_model(model, "m", sh, _d2_args(), torch.device("cpu"), None)
    assert no_d1["decoder"]["both_covered_rerank"] == {}


def test_d2_contrastive_model_gives_dual_rows_only(fake_coco):
    mod = _load_d2()
    sh = _d2_shared(mod, fake_coco)
    result, _ = mod.run_model(tiny_model("contrastive").eval(), "c", sh, _d2_args(), torch.device("cpu"), D1_JSON)
    assert "decoder" not in result and len(result["dual"]) == 10


def _assert_close(x, y, path=""):
    if isinstance(x, dict):
        assert x.keys() == y.keys(), path
        for k in x:
            _assert_close(x[k], y[k], f"{path}/{k}")
    elif x is None or y is None:
        assert x is None and y is None, path
    else:
        assert x == pytest.approx(y, abs=1e-4), path


@pytest.mark.parametrize("name,extra", [("fusion_multilearner", ()), ("fusion_none", ()),
                                        ("fusion_concat", ("model.mlm_image_source=clean",))])
def test_d2_results_do_not_depend_on_batch_size_or_pair_limit(name, extra, fake_coco):
    mod = _load_d2()
    model = tiny_model(name, *extra).eval()
    sh = _d2_shared(mod, fake_coco)
    big, _ = mod.run_model(model, "m", sh, _d2_args(batch_size=500), torch.device("cpu"), D1_JSON)
    small, _ = mod.run_model(model, "m", sh, _d2_args(batch_size=20), torch.device("cpu"), D1_JSON)
    big.pop("meta"), small.pop("meta")
    _assert_close(big, small)
    # a pair limit keeps the retained pairs' draws: pair 0..2 give the same balances
    sub = _d2_shared(mod, fake_coco, limit=3)
    _, full_arr = mod.run_model(model, "m", sh, _d2_args(), torch.device("cpu"), None)
    _, sub_arr = mod.run_model(model, "m", sub, _d2_args(), torch.device("cpu"), None)
    for key in full_arr:
        np.testing.assert_allclose(sub_arr[key], full_arr[key][:3], atol=1e-5)


def test_d2_fusion_none_selection_is_all_ties(fake_coco):
    mod = _load_d2()
    sh = _d2_shared(mod, fake_coco)
    result, _ = mod.run_model(tiny_model("fusion_none").eval(), "n", sh, _d2_args(), torch.device("cpu"), None)
    entry = result["decoder"]["selection"]["1"]["cA"]
    assert entry["n"] == 0 and entry["dropped"] == 5 and entry["mean"] is None


def test_d2_blend_inputs_are_the_models_normalised_tensors(fake_coco, monkeypatch):
    """Pure blends at lambda 1 reproduce image A: the condition builder feeds the loaded tensors unchanged."""
    mod = _load_d2()
    sh = _d2_shared(mod, fake_coco)
    idx = np.arange(5)
    images, ids = mod.condition_images(("blend", 1.0), sh["imgs_a"], sh["imgs_b"], idx)
    assert ids is None and torch.equal(images, sh["imgs_a"])
    mix, ids = mod.condition_images(("mix", 11), sh["imgs_a"], sh["imgs_b"], idx)
    assert ids.shape == (5, 13) and mix.shape == sh["imgs_a"].shape
