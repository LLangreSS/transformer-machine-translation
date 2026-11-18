
from pathlib import Path
from itertools import zip_longest

import torch

from src.data.tokenizer import Tokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "processed"
TOKENIZER_DIR = ROOT_DIR / "tokenizers"
OUTPUT_DIR = ROOT_DIR / "data" / "tokenized"

MAX_LENGTH = 256


def encode_split(split: str, en_tokenizer, ru_tokenizer):

    src_path = DATA_DIR / f"{split}.en.txt"
    tgt_path = DATA_DIR / f"{split}.ru.txt"

    src_sequences = []
    tgt_sequences = []

    skipped = 0

    with open(src_path, encoding="utf-8") as src, \
         open(tgt_path, encoding="utf-8") as tgt:

        for src_line, tgt_line in zip_longest(src, tgt):

            if src_line is None or tgt_line is None:
                raise ValueError("Source and target files have different lengths")

            src_ids = en_tokenizer.encode(src_line.strip())
            tgt_ids = ru_tokenizer.encode(tgt_line.strip())

            if len(src_ids) > MAX_LENGTH or len(tgt_ids) > MAX_LENGTH:
                skipped += 1
                continue

            src_sequences.append(src_ids)
            tgt_sequences.append(tgt_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "src_sequences": src_sequences,
            "tgt_sequences": tgt_sequences,
        },
        OUTPUT_DIR / f"{split}.pt"
    )

    print(
        f"{split}: {len(src_sequences)} pairs, "
        f"skipped: {skipped}"
    )


def main():

    en_tokenizer = Tokenizer(TOKENIZER_DIR / "en.model")
    ru_tokenizer = Tokenizer(TOKENIZER_DIR / "ru.model")

    for split in ["train", "valid", "test"]:
        encode_split(split, en_tokenizer, ru_tokenizer)


if __name__ == "__main__":
    main()
