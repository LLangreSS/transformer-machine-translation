
from pathlib import Path
import random
from itertools import zip_longest


ROOT_DIR = Path(__file__).resolve().parents[2]

SRC_FILE = ROOT_DIR / "data/raw/MultiUN.en-ru.en"
TGT_FILE = ROOT_DIR / "data/raw/MultiUN.en-ru.ru"

OUTPUT_DIR = ROOT_DIR / "data/processed"

SAMPLE_SIZE = 1_000_000
SEED = 42


def normalize_text(text):
    return " ".join(text.split())


def sample_parallel_data(src_file, tgt_file, sample_size, seed):

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    with open(src_file, encoding="utf-8") as src, \
         open(tgt_file, encoding="utf-8") as tgt:

        for src_line, tgt_line in zip_longest(src, tgt):

            if src_line is None or tgt_line is None:
                raise ValueError("Source and target files have different lengths")

            src_text = normalize_text(src_line)
            tgt_text = normalize_text(tgt_line)

            if not src_text or not tgt_text:
                continue

            seen += 1
            pair = (src_text, tgt_text)

            if len(reservoir) < sample_size:
                reservoir.append(pair)
            else:
                idx = rng.randrange(seen)

                if idx < sample_size:
                    reservoir[idx] = pair

    rng.shuffle(reservoir)

    return reservoir


def save_split(name, pairs):

    src_path = OUTPUT_DIR / f"{name}.en.txt"
    tgt_path = OUTPUT_DIR / f"{name}.ru.txt"

    with open(src_path, "w", encoding="utf-8") as src, \
         open(tgt_path, "w", encoding="utf-8") as tgt:

        for src_text, tgt_text in pairs:
            src.write(src_text + "\n")
            tgt.write(tgt_text + "\n")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = sample_parallel_data(
        SRC_FILE,
        TGT_FILE,
        SAMPLE_SIZE,
        SEED
    )

    train = pairs[:980_000]
    valid = pairs[980_000:990_000]
    test = pairs[990_000:1_000_000]

    save_split("train", train)
    save_split("valid", valid)
    save_split("test", test)

    print(f"Train: {len(train)}")
    print(f"Validation: {len(valid)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()
