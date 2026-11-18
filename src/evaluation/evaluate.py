import random
import torch
import sacrebleu

from src.config import Config
from src.model.transformer import Transformer
from src.data.tokenizer import Tokenizer


def load_model():
    model = Transformer(
        src_vocab_size=Config.SRC_VOCAB_SIZE,
        tgt_vocab_size=Config.TGT_VOCAB_SIZE,
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        num_encoder_layers=Config.NUM_ENCODER_LAYERS,
        num_decoder_layers=Config.NUM_DECODER_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT,
        src_padding_idx=Config.PAD_ID,
        tgt_padding_idx=Config.PAD_ID
    ).to(Config.DEVICE)

    checkpoint = torch.load(
        "checkpoints/last_model.pt",
        map_location=Config.DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    return model


def generate_translation(model, src_ids, max_length=Config.MAX_SEQ_LEN):
    src = torch.tensor(
        [src_ids],
        dtype=torch.long,
        device=Config.DEVICE
    )

    generated = [Config.BOS_ID]

    with torch.no_grad():
        for _ in range(max_length - 1):
            tgt = torch.tensor(
                [generated],
                dtype=torch.long,
                device=Config.DEVICE
            )

            logits = model(src, tgt)
            next_token = logits[0, -1].argmax(dim=-1).item()

            if next_token == Config.EOS_ID:
                break

            if next_token == Config.PAD_ID:
                break

            generated.append(next_token)

    return generated[1:]


def main():
    random.seed(Config.SEED)

    src_tokenizer = Tokenizer("tokenizers/en.model")
    tgt_tokenizer = Tokenizer("tokenizers/ru.model")

    model = load_model()

    test_data = torch.load(
        "data/tokenized/test.pt",
        map_location="cpu",
        weights_only=True
    )

    src_sequences = test_data["src_sequences"]
    tgt_sequences = test_data["tgt_sequences"]

    predictions = []
    references = []

    print(f"Evaluating {len(src_sequences)} test pairs...")

    for index, (src_ids, tgt_ids) in enumerate(
        zip(src_sequences, tgt_sequences), start=1
    ):
        src_ids = list(src_ids)
        tgt_ids = list(tgt_ids)

        prediction_ids = generate_translation(model, src_ids)

        prediction = tgt_tokenizer.decode(prediction_ids)
        reference = tgt_tokenizer.decode(
            [token for token in tgt_ids
             if token not in (Config.BOS_ID, Config.EOS_ID, Config.PAD_ID)]
        )

        predictions.append(prediction)
        references.append(reference)

        if index % 100 == 0:
            print(f"Processed {index}/{len(src_sequences)}", flush=True)

    bleu = sacrebleu.corpus_bleu(predictions, [references])

    print("\n" + "=" * 50)
    print(f"BLEU: {bleu.score:.2f}")
    print(f"Signature: {bleu}")
    print("=" * 50)

    print("\nRandom translation examples:\n")

    sample_indices = random.sample(
        range(len(predictions)),
        min(5, len(predictions))
    )

    for index in sample_indices:
        source = src_tokenizer.decode(
            [token for token in src_sequences[index]
             if token not in (Config.BOS_ID, Config.EOS_ID, Config.PAD_ID)]
        )

        print(f"EN: {source}")
        print(f"REF: {references[index]}")
        print(f"PRED: {predictions[index]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
