import numpy as np

from helpers import make_fake_al28
from mmae.data.artelingo import EMOTIONS, heldout_paintings
from mmae.engine.hb import data


def test_al28_targets_drop_english_merge_other_and_filter(tmp_path):
    csv = make_fake_al28(tmp_path / "al28.csv", {"p5": "Style_A/p5.jpg", "v2": "Style_A/v2.jpg"})
    paintings, counts = data.al28_targets(csv, min_votes=10)
    assert paintings.names == ["p5", "v2"] and paintings.images == ["Style_A/p5.jpg", "Style_A/v2.jpg"]
    # 25 votes each, every third English (v = 0, 3, ..., 24: 9 votes) -> 16 non-English; "sparse" has 5 -> dropped
    assert counts.sum(1).tolist() == [16, 16]
    assert counts[:, EMOTIONS.index("something else")].tolist() == [1, 1]  # the merged "other" vote


def test_val_labels_and_train_histograms_respect_the_holdout(fake_artelingo):
    _, annotations, heldout_file = fake_artelingo
    heldout = heldout_paintings(heldout_file)
    paintings, index, labels = data.val_labels(annotations, heldout)
    assert "v2" not in paintings.names and len(index) == len(labels) == 10
    train, counts = data.train_histograms(annotations, heldout)
    assert "p5" not in train.names and counts.sum() == 10
    prior = data.prior(annotations, heldout)
    assert prior.shape == (9,) and np.isclose(prior.sum(), 1.0)


def test_english_counts(fake_artelingo):
    _, annotations, _ = fake_artelingo
    counts = data.english_counts(["p1", "t0", "missing"], annotations)
    assert counts.sum(1).tolist() == [2, 5, 0]


def test_length_grid(fake_artelingo, tokenizer):
    _, annotations, heldout_file = fake_artelingo
    grid = data.length_grid(annotations, tokenizer, heldout_paintings(heldout_file), max_text_len=40)
    assert len(grid) == 9 and all(isinstance(x, int) and 1 <= x <= 38 for x in grid)
    assert grid == sorted(grid)


def test_test_captions(fake_artelingo):
    _, annotations, _ = fake_artelingo
    caps = data.test_captions(annotations)
    assert len(caps) == 15 and set(caps[0]) == {"painting", "image", "caption", "emotion"}
