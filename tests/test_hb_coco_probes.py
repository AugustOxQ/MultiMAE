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
