import pytest
import torch

from mmae.models.masking import random_patch_mask, random_token_mask


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_patch_mask_counts_and_complement():
    ids_keep, mask = random_patch_mask(4, 49, 0.75, generator=gen())
    assert ids_keep.shape == (4, 13) and mask.shape == (4, 49)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [36, 36, 36, 36]
    for b in range(4):
        assert torch.equal(ids_keep[b], (~mask[b]).nonzero().squeeze(1))


def test_patch_mask_differs_across_samples_and_is_reproducible():
    a = random_patch_mask(8, 49, 0.75, generator=gen(1))[1]
    b = random_patch_mask(8, 49, 0.75, generator=gen(1))[1]
    assert torch.equal(a, b)
    assert len({tuple(row.tolist()) for row in a}) == 8


@pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
def test_patch_mask_rejects_bad_ratio(ratio):
    with pytest.raises(ValueError):
        random_patch_mask(2, 49, ratio)


def token_batch():
    # CLIP layout: BOS words EOS PAD... where PAD == EOS and padding is flagged special.
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
    special = torch.tensor([[1, 0, 0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
    return attention, special


def test_token_mask_counts_and_never_special_or_padding():
    attention, special = token_batch()
    mask = random_token_mask(attention, special, 0.5, generator=gen())
    assert mask.dtype == torch.bool and mask.shape == attention.shape
    assert mask.sum(dim=1).tolist() == [2, 1, 0]  # round(0.5*4)=2, max(1, round(0.5))=1, nothing maskable
    assert not (mask & special.bool()).any()
    assert not (mask & ~attention.bool()).any()


def test_token_mask_masks_at_least_one_token():
    attention, special = token_batch()
    mask = random_token_mask(attention, special, 0.15, generator=gen())
    assert mask.sum(dim=1).tolist() == [1, 1, 0]


def test_token_mask_differs_across_samples_and_is_reproducible():
    attention = torch.tensor([[1] * 12] * 32)
    special = torch.tensor([[1] + [0] * 10 + [1]] * 32)
    a = random_token_mask(attention, special, 0.3, generator=gen(3))
    b = random_token_mask(attention, special, 0.3, generator=gen(3))
    assert torch.equal(a, b)
    assert len({tuple(row.tolist()) for row in a}) > 1


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_token_mask_rejects_bad_ratio(ratio):
    attention, special = token_batch()
    with pytest.raises(ValueError):
        random_token_mask(attention, special, ratio)
