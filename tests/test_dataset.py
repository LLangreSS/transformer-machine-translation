import torch
from torch.utils.data import DataLoader

from src.data.dataset import TranslationDataset, collate_fn


def test_dataloader():

    src_sequences = [
        [1, 15, 27, 2],
        [1, 8, 2],
        [1, 34, 56, 78, 2]
    ]

    tgt_sequences = [
        [1, 45, 67, 2],
        [1, 12, 2],
        [1, 89, 90, 91, 2]
    ]

    dataset = TranslationDataset(
        src_sequences,
        tgt_sequences
    )

    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=collate_fn
    )

    src, tgt = next(iter(dataloader))

    assert src.shape == (3, 5)
    assert tgt.shape == (3, 5)

    assert src.dtype == torch.long
    assert tgt.dtype == torch.long