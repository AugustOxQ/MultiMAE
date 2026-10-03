import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mmae.engine.diagnostics import (
    class_set_groups, count_class_words, drop_words, normalise_caption, neighbour_purity, per_query_rprecision, pmrp_rows,
    similarity_stats,
)

# 4 test images (COCO ids 10, 11, 12, 13), 2 captions each (ids 100..107); images 10 and 11 share a class set,
# 12 is alone, 13 has no PM entry.
IMAGE_IDS = np.array([10, 11, 12, 13])
CAPTION_IDS = np.array([[100, 101], [102, 103], [104, 105], [106, 107]])
PM = {"t2i": {100: [11], 101: [11], 102: [10], 103: [10], 104: [], 105: []}, "i2t": {10: [102, 103], 11: [100, 101]}}


def test_class_set_groups():
    groups = class_set_groups(PM, IMAGE_IDS, CAPTION_IDS)
    assert groups[0] == groups[1] and groups[2] not in (groups[0], -1) and groups[3] == -1


def test_neighbour_purity():
    # nearest other item: 0 -> 1, 1 -> 0, 2 -> 3; item 3 is alone in group 1 (no mate), so it is not a query
    emb = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.6, 0.4]]), dim=-1)
    groups = np.array([0, 0, 0, 1])
    assert neighbour_purity(emb, groups, k=1) == pytest.approx(200.0 / 3)
    # owners: items 0 and 1 share an owner, so neither may pick the other; both then pick item 3 (group 1)
    assert neighbour_purity(emb, groups, k=1, owners=np.array([0, 0, 1, 2])) == 0.0


def test_pmrp_rows_and_rprecision():
    rows = pmrp_rows(PM, IMAGE_IDS, CAPTION_IDS)
    queries, positives = rows["t2i"]
    assert queries.tolist() == [0, 1, 2, 3, 4, 5]
    assert sorted(positives[0].tolist()) == [0, 1]           # own image 10 (row 0) + image 11 (row 1)
    image_emb = torch.eye(4)
    # captions of images 10 and 11 rank their own image first and image 12 (row 2, not a positive) second;
    # captions of image 12 (one positive) rank it first
    caption_emb = torch.tensor([[1.0, 0, 0.5, 0], [1.0, 0, 0.5, 0], [0, 1.0, 0.5, 0], [0, 1.0, 0.5, 0],
                                [0, 0, 1.0, 0.5], [0, 0, 1.0, 0.5]])
    r = per_query_rprecision(caption_emb, image_emb, positives, max_r=50)
    assert r.tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0, 1.0])


def test_similarity_stats():
    image_emb = torch.eye(3)
    caption_emb = torch.eye(3)[:, None, :].repeat(1, 2, 1)   # every caption equals its image
    stats = similarity_stats(image_emb, caption_emb, np.array([0, 0, 1]))
    assert stats == {"pos": 1.0, "same_class_neg": 0.0, "other_neg": 0.0}


def test_class_words_and_drop_words():
    assert count_class_words("A man riding a horse next to two dogs.") == 3  # man (person), horse, dogs
    rng = random.Random(0)
    assert drop_words("a dog on the beach", "content", 1, rng) in {"a on the beach", "a dog on the"}
    assert drop_words("a dog on the beach", "stop", 3, rng) == "dog beach"
    assert drop_words("a dog", "content", 2, rng) is None


@pytest.mark.parametrize("caption, expected", [
    ("A man riding a horse next to two dogs.", 3),
    ("a traffic light and a stop sign", 2),
    ("a hot dog on a plate", 1),
    ("an orange cat on a table", 2),
    ("two dogs and a dog", 1),
    ("a teddy bear and two bears", 2),
])
def test_count_class_words_counts_distinct_classes(caption, expected):
    assert count_class_words(caption) == expected


def test_normalise_caption_matches_the_text_drop_words_rebuilds():
    assert normalise_caption("A dog.  On the beach.") == "A dog On the beach"
    caption = "A dog.  On the beach."
    assert drop_words(caption, "stop", 0, random.Random(0)) == normalise_caption(caption)
