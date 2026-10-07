"""ArtELingo data for H-b (spec 2026-10-07, section 4)."""
from pathlib import Path

import pytest
import torch

from helpers import ARTELINGO_EMOTIONS
from mmae.data.artelingo import (
    EMOTIONS, HELDOUT_FILE, ArtelingoPairs, ArtelingoRetrieval, heldout_paintings, load_painting, retrieval_groups,
)
from mmae.data.transforms import build_image_transform
from mmae.models.model import NUM_EMOTIONS

AL28_CSV = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv")


@pytest.fixture(scope="module")
def transform():
    return build_image_transform("openai/clip-vit-base-patch32")


def test_emotion_classes_match_the_model():
    assert EMOTIONS == ARTELINGO_EMOTIONS and len(EMOTIONS) == NUM_EMOTIONS


def test_packaged_heldout_list_is_every_al28_painting():
    paintings = heldout_paintings()
    assert len(paintings) == 1658
    if AL28_CSV.exists():
        import pandas as pd
        assert paintings == frozenset(pd.read_csv(AL28_CSV, usecols=["painting"])["painting"])


def test_train_pairs_drop_heldout_paintings(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    pairs = ArtelingoPairs(images, annotations, "train", transform, heldout=heldout_paintings(heldout_file))
    assert len(pairs) == 10  # 6 paintings x 2 captions, p5 held out
    assert not any(caption.startswith("p5 ") for _, caption, _ in pairs.pairs)
    image, caption, emotion = pairs[0]
    assert image.shape == (3, 224, 224) and isinstance(caption, str) and 0 <= emotion < 9


def test_val_pairs_are_caption_major_and_drop_heldout(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    pairs = ArtelingoPairs(images, annotations, "val", transform, heldout=heldout_paintings(heldout_file))
    paintings = [caption.split()[0] for _, caption, _ in pairs.pairs]
    assert paintings == ["v0", "v1"] * 5  # caption-major, v2 held out
    assert [e for _, c, e in pairs.pairs if c.startswith("v1 ")] == [EMOTIONS.index(ARTELINGO_EMOTIONS[(1 + k) % 9]) for k in range(5)]


def test_retrieval_drops_heldout_from_val_but_not_test(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    heldout = heldout_paintings(heldout_file)
    val = ArtelingoRetrieval(images, annotations, "val", transform, heldout=heldout)
    test = ArtelingoRetrieval(images, annotations, "test", transform, heldout=heldout)
    assert len(val) == 2 and len(test) == 3
    image, captions = test[1]
    assert image.shape == (3, 224, 224) and len(captions) == 5 and captions[0].startswith("t1 ")
    assert test.emotions[1] == [EMOTIONS.index(ARTELINGO_EMOTIONS[(1 + k) % 9]) for k in range(5)]


def test_retrieval_caption_without_emotion_fails(fake_artelingo):
    images, annotations, _ = fake_artelingo
    path = annotations / "artelingo_test.json"
    import json
    items = json.loads(path.read_text())
    path.write_text(json.dumps(items[1:]))  # drop t0's first caption from the per-caption file
    with pytest.raises(KeyError, match="t0"):
        retrieval_groups(annotations, "test")


def test_limit_keeps_the_first_items(fake_artelingo, transform):
    images, annotations, heldout_file = fake_artelingo
    assert len(ArtelingoPairs(images, annotations, "train", transform, limit=3)) == 3
    assert len(ArtelingoRetrieval(images, annotations, "test", transform, limit=2)) == 2


def test_load_painting_returns_rgb_at_least_224(fake_artelingo):
    images, _, _ = fake_artelingo
    image = load_painting(images / "Style_A" / "p1.jpg")  # grayscale on disk
    assert image.mode == "RGB" and min(image.size) >= 224


@pytest.mark.slow
def test_real_artelingo_splits():
    """The real files: hold-out counts and the retrieval lookups (spec section 4)."""
    annotations = Path("/data/PDD/artelingo")
    heldout = heldout_paintings()
    val = retrieval_groups(annotations, "val", heldout)
    test = retrieval_groups(annotations, "test", heldout)
    assert len(test) == 4975  # lookups complete (retrieval_groups raises otherwise)
    assert len(val) == 2421   # 2,469 val retrieval paintings minus 48 AL-28 paintings
    assert not any(Path(img).stem in heldout for img, _, _ in val)
    train = ArtelingoPairs(annotations.parent / "wikiart_proj" / "wikiart", annotations, "train", lambda x: x, heldout=heldout)
    assert len(train) == 302841  # 308,723 captions minus those of the 1,160 AL-28 train paintings
