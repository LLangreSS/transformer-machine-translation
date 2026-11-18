
from pathlib import Path

import sentencepiece as spm


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "tokenizers"


def train_tokenizer(input_file: Path, model_prefix: Path):
    spm.SentencePieceTrainer.train(
        input=str(input_file),
        model_prefix=str(model_prefix),
        vocab_size=24_000,
        model_type="unigram",

        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",

        character_coverage=0.9995,
        hard_vocab_limit=False,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_tokenizer(
        DATA_DIR / "train.en.txt",
        OUTPUT_DIR / "en"
    )

    train_tokenizer(
        DATA_DIR / "train.ru.txt",
        OUTPUT_DIR / "ru"
    )

    print("Tokenizers trained successfully!")


if __name__ == "__main__":
    main()
