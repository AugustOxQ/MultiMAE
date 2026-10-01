"""Pytest fixtures shared by all tests."""
import pytest

from helpers import CLIP_NAME


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CLIP_NAME)
