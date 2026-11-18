
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TranslationDataset, collate_fn


ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    data = torch.load(
        ROOT_DIR / "data/tokenized/train.pt",
        map_location="cpu",
        weights_only=True
    )

    dataset = TranslationDataset(
        data["src_sequences"],
        data["tgt_sequences"]
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    src, tgt = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Source shape:", src.shape)
    print("Target shape:", tgt.shape)
    print("Source dtype:", src.dtype)
    print("Target dtype:", tgt.dtype)

    print("\nFirst source sequence:", src[0])
    print("First target sequence:", tgt[0])


if __name__ == "__main__":
    main()
