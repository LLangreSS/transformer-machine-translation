import torch

from src.model.masks import generate_causal_mask, generate_padding_mask, generate_decoder_mask


def test_causal_mask():

    mask = generate_causal_mask(4)

    expected = torch.tensor([
        [False, True,  True,  True],
        [False, False, True,  True],
        [False, False, False, True],
        [False, False, False, False]
    ])

    assert torch.equal(mask, expected)


def test_padding_mask():

    sequence = torch.tensor([
        [5, 8, 12, 0, 0],
        [3, 7, 9, 4, 0]
    ])

    mask = generate_padding_mask(sequence, padding_idx=0)

    expected = torch.tensor([
        [[[False, False, False, True, True]]],
        [[[False, False, False, False, True]]]
    ])

    assert torch.equal(mask, expected)
    assert mask.shape == (2, 1, 1, 5)

def test_decoder_mask():

    sequence = torch.tensor([
        [5, 8, 0]
    ])

    mask = generate_decoder_mask(sequence, padding_idx=0)

    expected = torch.tensor([
        [
            [
                [False, True, True],
                [False, False, True],
                [False, False, True]
            ]
        ]
    ])

    assert torch.equal(mask, expected)
    assert mask.shape == (1, 1, 3, 3)