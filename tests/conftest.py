"""Pytest fixtures shared by all tests."""
import pytest

from helpers import CLIP_NAME, make_fake_coco


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CLIP_NAME)


@pytest.fixture
def fake_coco(tmp_path):
    return make_fake_coco(tmp_path / "coco")
