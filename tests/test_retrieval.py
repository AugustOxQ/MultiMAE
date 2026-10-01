import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

import _ddp_workers
from helpers import run_ddp_worker
from mmae.engine.retrieval import encode_retrieval_set, evaluate_retrieval, retrieval_metrics
from reference_v0_retrieval import v0_metrics


def random_embeddings(n: int = 50, k: int = 5, d: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    captions = F.normalize(images.unsqueeze(1) + 0.9 * torch.randn(n, k, d, generator=g), dim=-1)
    return images, captions


def v0_on(images, captions):
    n, k, _ = captions.shape
    text_to_image = torch.arange(n).repeat_interleave(k)
    image_to_text = torch.arange(n * k).view(n, k)
    return v0_metrics(images, captions.reshape(n * k, -1), text_to_image, image_to_text)


def test_metrics_match_v0():
    images, captions = random_embeddings()
    new, old = retrieval_metrics(images, captions, chunk=7), v0_on(images, captions)
    for key, value in old.items():
        if key.endswith(("meanR", "medR")):
            assert int(round(new[key])) == value, key
        else:
            assert abs(new[key] - value) < 0.011, (key, new[key], value)
    assert abs(new["rsum"] - sum(new[f"{d}_R{k}"] for d in ("i2t", "t2i") for k in (1, 5, 10))) < 1e-9


def test_perfect_embeddings_give_full_recall():
    images, _ = random_embeddings(n=20)
    metrics = retrieval_metrics(images, images.unsqueeze(1).repeat(1, 5, 1))
    assert metrics["i2t_R1"] == 100.0 and metrics["t2i_R1"] == 100.0 and metrics["rsum"] == 600.0
    assert metrics["t2i_meanR"] == 1.0 and metrics["t2i_mAP"] == 100.0
    # tied positives take distinct sorted positions 1..5, as in v0's argsort
    assert metrics["i2t_mAP"] == 100.0
    assert metrics["i2t_meanR"] == 3.0


def single_process_reference():
    model = _ddp_workers.DummyEmbedder()
    loader = DataLoader(_ddp_workers.retrieval_dataset(), batch_size=8, collate_fn=default_collate)
    model.train()
    images, captions = encode_retrieval_set(model, loader)
    assert model.training  # mode restored
    return images, captions, evaluate_retrieval(model, loader)


def test_two_processes_match_one(tmp_path):
    result = run_ddp_worker("retrieval", tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    two = torch.load(tmp_path / "retrieval.pt")
    images, captions, metrics = single_process_reference()
    assert two["images"].shape == (43, 8) and two["captions"].shape == (43, 5, 8)
    assert torch.allclose(two["images"], images, atol=1e-6) and torch.allclose(two["captions"], captions, atol=1e-6)
    # matmuls over differently sized batches may differ in the last bit, so compare approximately
    assert two["metrics"] == pytest.approx(metrics)


def test_unprepared_loader_raises_under_two_processes(tmp_path):
    result = run_ddp_worker("retrieval", tmp_path, 2, "--unprepared")
    assert result.returncode == 0, result.stderr[-3000:]
    assert (tmp_path / "raised0").exists() and (tmp_path / "raised1").exists()
