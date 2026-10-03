import pytest
import torch
from torch.utils.data import DataLoader

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


def test_val_flattens_first_five_captions_caption_major(fake_coco, transform):
    pairs = CocoPairs(*fake_coco, "val", transform)
    assert len(pairs) == 6 * CAPTIONS_PER_IMAGE
    # every image with its first caption, then every image with its second caption, ...; image 0's 6th caption is dropped
    assert pairs.pairs == [(f"val/val_{i}.jpg", f"val {i} caption {k}") for k in range(5) for i in range(6)]
    retrieval = CocoRetrieval(*fake_coco, "val", transform)
    assert len(retrieval) == 6
    image, captions = retrieval[0]
    assert image.shape == (3, 224, 224) and captions == [f"val 0 caption {k}" for k in range(5)]


def test_limit_takes_first_items_and_clamps(fake_coco, transform):
    assert len(CocoPairs(*fake_coco, "train", transform, limit=3)) == 3
    assert len(CocoPairs(*fake_coco, "train", transform, limit=1000)) == 16
    assert len(CocoRetrieval(*fake_coco, "test", transform, limit=2)) == 2
    # limit N on val/test pairs (N <= images) = the first N images with their first caption = CocoRetrieval's N images
    pairs, retrieval = CocoPairs(*fake_coco, "test", transform, limit=4), CocoRetrieval(*fake_coco, "test", transform, limit=4)
    assert [image for image, _ in pairs.pairs] == [image for image, _ in retrieval.items]
    assert [caption for _, caption in pairs.pairs] == [captions[0] for _, captions in retrieval.items]


@pytest.mark.parametrize("split", ["val", "test"])
def test_eval_batches_hold_distinct_images(fake_coco, transform, split):
    """An unshuffled eval batch of B <= 6 (the number of images) consecutive pairs never repeats an image, so
    no caption in it is a false negative for the contrastive loss, including across caption rounds."""
    pairs = CocoPairs(*fake_coco, split, transform)
    images = [image for image, _ in pairs.pairs]
    for batch_size in range(1, 7):
        first = next(iter(DataLoader(range(len(pairs)), batch_size=batch_size, shuffle=False)))
        assert len({images[i] for i in first.tolist()}) == batch_size
        for start in range(len(images) - batch_size + 1):
            assert len(set(images[start : start + batch_size])) == batch_size, (batch_size, start)


def test_short_caption_lists_are_rejected(tmp_path, transform):
    import json

    (tmp_path / "coco_karpathy_val.json").write_text(json.dumps([{"image": "a.jpg", "caption": ["x"] * 4}]))
    for dataset in (CocoPairs, CocoRetrieval):
        with pytest.raises(ValueError, match="fewer than 5 captions"):
            dataset(tmp_path, tmp_path, "val", transform)


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


def test_collator_marks_content_tokens(tmp_path):
    from mmae.data import Collator
    from helpers import CLIP_NAME

    collator = Collator(CLIP_NAME, 16, content_words=True)
    batch = collator.tokenize(["a dog is running on the beach .", "the of and"])
    content = batch["content_tokens_mask"]
    words = [collator.tokenizer.convert_ids_to_tokens(row) for row in batch["input_ids"].tolist()]
    marked = [[w for w, c in zip(ws, cs) if c] for ws, cs in zip(words, content.tolist())]
    assert marked[0] == ["dog</w>", "running</w>", "beach</w>"]
    assert marked[1] == []
    assert not (content & ~batch["attention_mask"].bool()).any()
    assert not (content & batch["special_tokens_mask"].bool()).any()
    assert "content_tokens_mask" not in Collator(CLIP_NAME, 16).tokenize(["a dog"])
