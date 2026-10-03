import json
import os
import subprocess
import sys

from helpers import REPO, make_fake_vwsd, run_train


def test_diagnose_on_two_tiny_runs(tmp_path, fake_coco):
    for model in ("contrastive", "fusion_concat"):
        result = run_train(tmp_path, fake_coco, f"model={model}", "train.save=best")
        assert result.returncode == 0, result.stderr[-3000:]
    runs_root = next((tmp_path / "res").rglob("checkpoints")).parents[1]
    images_dir, annotations_dir = fake_coco
    out = tmp_path / "diag"
    cmd = [sys.executable, str(REPO / "scripts" / "diagnose.py"),
           f"data.images_dir={images_dir}", f"data.annotations_dir={annotations_dir}",
           "model.backbone.pretrained=tiny-random-clip", "train=debug", "train.num_workers=0",
           "eval.extended_metrics=false", f"eval.vwsd_dir={make_fake_vwsd(tmp_path / 'vwsd')}",
           f"+diag.runs_root={runs_root}", f"+diag.out={out}", "+diag.probe_captions=6"]
    result = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=1200,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "disabled"})
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads((out / "diagnostics.json").read_text())
    assert set(report["runs"]) >= {"zeroshot"} and len(report["runs"]) == 3
    for info in report["runs"].values():
        assert "vwsd/hit1" in info["vwsd"] and "probe" in info
    assert any("img+contrastive_txt" in key for key in report["swaps"])
    assert len(list((out / "embeddings").glob("*.pt"))) == 3
