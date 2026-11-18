import random
import torch

from src.config import Config
from src.data.tokenizer import Tokenizer
from src.evaluation.translate import load_model, translate


def main():
    random.seed(42)

    en = Tokenizer("tokenizers/en.model")
    ru = Tokenizer("tokenizers/ru.model")

    model = load_model()

    for split in ["train", "test"]:
        data = torch.load(
            f"data/tokenized/{split}.pt",
            map_location="cpu",
            weights_only=True
        )

        indices = random.sample(
            range(len(data["src_sequences"])),
            10
        )

        print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

        for i in indices:
            src_ids = [
                x for x in data["src_sequences"][i]
                if x != Config.PAD_ID
            ]

            tgt_ids = [
                x for x in data["tgt_sequences"][i]
                if x not in (
                    Config.PAD_ID,
                    Config.BOS_ID,
                    Config.EOS_ID
                )
            ]

            source = en.decode(src_ids)
            reference = ru.decode(tgt_ids)
            prediction = translate(source, model, en, ru)

            print(f"\nEN:   {source}")
            print(f"REF:  {reference}")
            print(f"PRED: {prediction}")
            print("-" * 60)


if __name__ == "__main__":
    main()