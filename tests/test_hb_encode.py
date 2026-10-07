import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from helpers import compose_cfg, make_fake_al28
from mmae.data.transforms import build_image_transform
from mmae.engine.hb import encode
from mmae.models import MultiMAE
from mmae.models.backbones import TINY_CLIP

ML80 = ("model=fusion_multilearner", "model.emotion_head=true", "model.masking.text_ratio=0.8", "model.loss.weights.mae=0")
PARCAP = ("model=fusion_multilearner", "model.emotion_head=true", "model.masking.text_ratio=1.0",
          "model.mlm_image_source=clean", "model.loss.weights.mae=0")
ROOT = Path(__file__).resolve().parents[1]


def make_run(tmp_path, name, *overrides):
    torch.manual_seed(0)
    cfg = compose_cfg(*overrides, f"model.backbone.pretrained={TINY_CLIP}", "data=artelingo")
    model = MultiMAE(cfg.model, max_text_len=cfg.data.max_text_len)
    run = tmp_path / name
    (run / "checkpoints").mkdir(parents=True)
    OmegaConf.save(cfg, run / "config.yaml")
    torch.save({"model": model.state_dict()}, run / "checkpoints" / "best.pt")
    return run


def sensitive(model):
    """Random init leaves the emotion logits almost input-independent; spread them so a mix-up shows."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


def test_view_ids_are_fixed_and_distinct():
    a, b = encode.view_ids(5, 3), encode.view_ids(5, 3)
    assert a.shape == (3, 5, 13) and torch.equal(a, b) and not torch.equal(a[0], a[1])


def test_hidden_captions_hide_every_real_token(tokenizer):
    batch = encode.hidden_captions(tokenizer, [1, 5, 38], 40)
    assert batch["token_mask"].sum(1).tolist() == [1, 5, 38]
    assert batch["attention_mask"].sum(1).tolist() == [3, 7, 40]


@pytest.mark.parametrize("arm", ["ml80", "parcap", "contrastive"])
def test_encode_paintings_shapes(arm, tmp_path, fake_artelingo, tokenizer):
    images_dir, _, _ = fake_artelingo
    overrides = {"ml80": ML80, "parcap": PARCAP, "contrastive": ("model=contrastive",)}[arm]
    model, _ = encode.load_run(make_run(tmp_path, arm, *overrides), "cpu")
    paths = ["Style_A/p0.jpg", "Style_A/p1.jpg", "Style_A/v0.jpg"]
    out = encode.encode_paintings(model, paths, images_dir, build_image_transform("openai/clip-vit-base-patch32"),
                                  encode.hidden_captions(tokenizer, [2, 4], 40), encode.view_ids(3, 4), 2, "cpu",
                                  workers=0)
    assert out["emb_full"].shape[0] == 3 and out["emb_views"].shape[:2] == (3, 4)
    if arm == "contrastive":
        assert "dec_views" not in out
    else:
        assert out["dec_views"].shape == (3, 4, 2, 9) and out["dec_full"].shape == (3, 1, 2, 9)
        assert np.isfinite(out["dec_views"].astype(np.float32)).all()


def test_batched_lengths_match_one_pass_per_length(tmp_path, fake_artelingo, tokenizer):
    """The length axis is batched into one decoder pass; each (image, length) must equal its own pass."""
    images_dir, _, _ = fake_artelingo
    model = sensitive(encode.load_run(make_run(tmp_path, "ml80", *ML80), "cpu")[0])
    transform = build_image_transform("openai/clip-vit-base-patch32")
    hidden = encode.hidden_captions(tokenizer, [2, 4, 7], 40)
    paths = ["Style_A/p0.jpg", "Style_A/p1.jpg", "Style_A/p2.jpg"]
    views = encode.view_ids(3, 2)
    out = encode.encode_paintings(model, paths, images_dir, transform, hidden, views, 2, "cpu", workers=0)
    from mmae.data.artelingo import load_painting
    with torch.no_grad():
        for i in (0, 2):  # item 2 sits alone in the second batch
            image = transform(load_painting(Path(images_dir) / paths[i])).unsqueeze(0)
            for v in range(2):
                for l in range(3):
                    _, emo = model.decode_text(image, hidden["input_ids"][l:l + 1], hidden["attention_mask"][l:l + 1],
                                               hidden["token_mask"][l:l + 1], views[v, i:i + 1])
                    np.testing.assert_allclose(out["dec_views"][i, v, l].astype(np.float32), emo[0].numpy(), atol=2e-3, rtol=2e-3)
            _, emo = model.decode_text(image, hidden["input_ids"][1:2], hidden["attention_mask"][1:2],
                                       hidden["token_mask"][1:2], None)
            np.testing.assert_allclose(out["dec_full"][i, 0, 1].astype(np.float32), emo[0].numpy(), atol=2e-3, rtol=2e-3)


def test_encode_d7(tmp_path, fake_artelingo, tokenizer):
    images_dir, annotations, _ = fake_artelingo
    from mmae.engine.hb.data import test_captions
    model, _ = encode.load_run(make_run(tmp_path, "ml80", *ML80), "cpu")
    caps = test_captions(annotations)[:4]
    out = encode.encode_d7(model, caps, images_dir, build_image_transform("openai/clip-vit-base-patch32"),
                           tokenizer, 40, 2, 4, "cpu", workers=0)
    assert out["d7_real"].shape == (4, 9, 2, 9) and out["d7_null"].shape == (4, 9, 2, 9)
    assert out["d7_valid"][:, 0].all() and out["d7_label"].shape == (4,)


def test_encode_d7_clean_source_uses_the_full_image(tmp_path, fake_artelingo, tokenizer):
    images_dir, annotations, _ = fake_artelingo
    from mmae.engine.hb.data import test_captions
    model, _ = encode.load_run(make_run(tmp_path, "parcap", *PARCAP), "cpu")
    out = encode.encode_d7(model, test_captions(annotations)[:3], images_dir,
                           build_image_transform("openai/clip-vit-base-patch32"), tokenizer, 40, 2, 4, "cpu", workers=0)
    assert out["d7_real"].shape == (3, 9, 1, 9) and out["d7_null"].shape == (3, 9, 1, 9)


@pytest.mark.parametrize("arm", ["ml80", "parcap"])
def test_d7_matches_single_decode_calls(arm, tmp_path, fake_artelingo, tokenizer):
    """Batched D7 (batch 2, 3 captions, patterns batched) equals one decode_text call per (caption, pattern, view),
    for the real image and for the null image through the same views."""
    from mmae.data.artelingo import load_painting
    from mmae.engine.hb.data import test_captions
    images_dir, annotations, _ = fake_artelingo
    model = sensitive(encode.load_run(make_run(tmp_path, arm, *(ML80 if arm == "ml80" else PARCAP)), "cpu")[0])
    transform = build_image_transform("openai/clip-vit-base-patch32")
    caps = test_captions(annotations)[::5]  # three different paintings
    out = encode.encode_d7(model, caps, images_dir, transform, tokenizer, 40, 2, 2, "cpu", workers=0)
    masked = arm == "ml80"
    n_views = 2 if masked else 1
    assert out["d7_real"].shape == (3, 9, n_views, 9) == out["d7_null"].shape
    enc = tokenizer([c["caption"] for c in caps], max_length=40, truncation=True, padding="max_length",
                    return_attention_mask=True, return_special_tokens_mask=True, return_tensors="pt")
    real = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
    from mmae.data.stopwords import content_token_table
    content = content_token_table(tokenizer)[enc["input_ids"]]
    gen = torch.Generator().manual_seed(encode.D7_SEED)
    views = encode.view_ids(3, 2, seed=5678)
    checked = 0
    with torch.no_grad():
        for i, cap in enumerate(caps):
            image = transform(load_painting(Path(images_dir) / cap["image"])).unsqueeze(0)
            for k, pattern in enumerate(encode.D7_PATTERNS):
                visible = encode.d7_visible(real[i], content[i], pattern, gen)
                tm = real[i] & ~visible if visible is not None else real[i]
                assert bool(out["d7_valid"][i, k]) == (visible is not None)
                for v in range(n_views):
                    keep = views[v, i:i + 1] if masked else None
                    for name, img in (("d7_real", image), ("d7_null", torch.zeros_like(image))):
                        _, emo = model.decode_text(img, enc["input_ids"][i:i + 1], enc["attention_mask"][i:i + 1],
                                                   tm.unsqueeze(0), keep)
                        np.testing.assert_allclose(out[name][i, k, v].astype(np.float32), emo[0].numpy(), atol=2e-3, rtol=2e-3)
                        checked += 1
    assert checked == 3 * 9 * n_views * 2


def test_script_refuses_missing_images(tmp_path, fake_artelingo):
    images_dir, annotations, _ = fake_artelingo
    run = make_run(tmp_path, "r", *ML80)
    csv_path = make_fake_al28(tmp_path / "al28.csv", {"p0": "Style_A/p0.jpg", "gone": "Style_A/gone.jpg"})
    cmd = [sys.executable, str(ROOT / "scripts" / "hb_encode.py"), "--runs", str(run), "--out", str(tmp_path / "o"),
           "--images-dir", str(images_dir), "--annotations-dir", str(annotations), "--al28-csv", str(csv_path),
           "--min-votes", "10", "--workers", "0"]
    result = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and "1 image files missing" in result.stderr and "gone.jpg" in result.stderr
    assert not (tmp_path / "o" / "r").exists()


def test_script_contrastive_meta_has_nulls(tmp_path, fake_artelingo):
    images_dir, annotations, _ = fake_artelingo
    run = make_run(tmp_path, "c", "model=contrastive")
    csv_path = make_fake_al28(tmp_path / "al28.csv", {"p0": "Style_A/p0.jpg", "p1": "Style_A/p1.jpg"})
    cmd = [sys.executable, str(ROOT / "scripts" / "hb_encode.py"), "--runs", str(run), "--out", str(tmp_path / "o"),
           "--images-dir", str(images_dir), "--annotations-dir", str(annotations), "--al28-csv", str(csv_path),
           "--min-votes", "10", "--workers", "0", "--views", "2", "--batch-size", "2"]
    result = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    meta = json.loads((tmp_path / "o" / "c" / "meta.json").read_text())
    assert meta["mlm_image_source"] is None and meta["text_ratio"] is None and meta["mae_weight"] is None


def test_d7_visible_patterns():
    real = torch.tensor([False, True, True, True, True, True, False])
    content = torch.tensor([False, True, False, True, True, False, False])
    g = torch.Generator().manual_seed(0)
    assert not encode.d7_visible(real, content, "j0", g).any()
    assert encode.d7_visible(real, content, "prefix2", g).nonzero().flatten().tolist() == [1, 3]
    assert encode.d7_visible(real, content, "random2", g).sum() == 2
    assert encode.d7_visible(real, content, "prefix4", g) is None


def test_script_smoke(tmp_path, fake_artelingo):
    images_dir, annotations, _ = fake_artelingo
    run = make_run(tmp_path, "20261007_000000_ml80", *ML80)
    csv_path = make_fake_al28(tmp_path / "al28.csv", {"p0": "Style_A/p0.jpg", "p1": "Style_A/p1.jpg"})
    out = tmp_path / "out"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    cmd = [sys.executable, str(ROOT / "scripts" / "hb_encode.py"), "--runs", str(run), "--out", str(out),
           "--images-dir", str(images_dir), "--annotations-dir", str(annotations), "--al28-csv", str(csv_path),
           "--batch-size", "2", "--views", "2", "--d7-views", "2", "--workers", "0", "--min-votes", "10"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    folder = out / run.name
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["mlm_image_source"] == "masked" and meta["text_ratio"] == 0.8 and meta["mae_weight"] == 0
    assert len(meta["lengths"]) > 0 and meta["al28"] and meta["val"] and meta["train"]
    z = np.load(folder / "encode.npz")
    n_len = len(meta["lengths"])
    assert z["al28_dec_views"].shape == (len(meta["al28"]), 2, n_len, 9)
    assert z["train_emb"].shape[0] == len(meta["train"]) and z["prompts"].shape[0] == 9
    assert z["d7_real"].shape[2] == 2 and "logit_scale" in z.files
    assert (out / "length_grid.json").is_file()
