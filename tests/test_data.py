import pytest
import torch

from helpers import CLIP_NAME
from mmae.data import CAPTIONS_PER_IMAGE, Collator, CocoPairs, CocoRetrieval, build_image_transform

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@pytest.fixture(scope="module")
def transform():
    return build_image_transform(CLIP_NAME)


def test_transform_matches_clip_preprocessing(transform):
    from PIL import Image

    out = transform(Image.new("RGB", (320, 240), (255, 0, 128)))
    assert out.shape == (3, 224, 224)
    expected = [(c / 255 - m) / s for c, m, s in zip((255, 0, 128), CLIP_MEAN, CLIP_STD)]
    assert torch.allclose(out[:, 100, 100], torch.tensor(expected), atol=1e-2)


def test_train_pairs(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "train", transform)
    assert len(pairs) == 16
    image, caption = pairs[0]
    assert image.shape == (3, 224, 224) and caption == "train caption 0 0"


def test_odd_images_load_as_rgb(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "train", transform)
    for index in (2, 4, 6):  # grayscale, CMYK, truncated JPEG
        image, _ = pairs[index]
        assert image.shape == (3, 224, 224) and torch.isfinite(image).all()


def test_val_flattens_first_five_captions(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "val", transform)
    assert len(pairs) == 6 * CAPTIONS_PER_IMAGE
    assert [pairs.pairs[i][1] for i in range(5)] == [f"val 0 caption {k}" for k in range(5)]
    retrieval = CocoRetrieval(*fake_coco, "val", transform)
    assert len(retrieval) == 6
    image, captions = retrieval[0]
    assert image.shape == (3, 224, 224) and captions == [f"val 0 caption {k}" for k in range(5)]


def test_limit_takes_first_items_and_clamps(fake_coco, transform):
    assert len(CocoPairs(*fake_coco, "train", transform, limit=3)) == 3
    assert len(CocoPairs(*fake_coco, "train", transform, limit=1000)) == 16
    assert len(CocoRetrieval(*fake_coco, "test", transform, limit=2)) == 2


def test_bad_split(fake_coco, transform):
    with pytest.raises(ValueError):
        CocoPairs(*fake_coco, "dev", transform)
    with pytest.raises(ValueError):
        CocoRetrieval(*fake_coco, "train", transform)


def test_collator_pairs_and_truncation_keeps_eos(tokenizer):
    collate = Collator(CLIP_NAME, max_text_len=8)
    batch = collate.pairs([(torch.zeros(3, 224, 224), "a"), (torch.ones(3, 224, 224), "word " * 30)])
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["special_tokens_mask"].shape == (2, 8)
    assert batch["input_ids"][1, -1].item() == tokenizer.eos_token_id  # truncated caption still ends with EOS
    assert batch["attention_mask"][1].all()
    assert batch["special_tokens_mask"][0].tolist() == [1, 0, 1, 1, 1, 1, 1, 1]  # BOS a EOS PAD...


def test_collator_retrieval_shapes():
    collate = Collator(CLIP_NAME, max_text_len=8)
    batch = collate.retrieval([(torch.zeros(3, 224, 224), [f"c {k}" for k in range(5)])] * 3)
    assert batch["pixel_values"].shape == (3, 3, 224, 224)
    assert batch["input_ids"].shape == batch["attention_mask"].shape == (3, 5, 8)


@pytest.mark.slow
def test_real_coco_sizes(transform):
    root, ann = "/data/SSD/coco/images", "/data/SSD/coco/annotations"
    assert len(CocoPairs(root, ann, "train", transform)) == 566747
    assert len(CocoPairs(root, ann, "val", transform)) == 25000
    assert len(CocoRetrieval(root, ann, "test", transform)) == 5000
    image, captions = CocoRetrieval(root, ann, "test", transform)[0]
    assert image.shape == (3, 224, 224) and len(captions) == 5
