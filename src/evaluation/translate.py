import torch

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
        "checkpoints/best_model.pt",
        map_location=Config.DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Device: {Config.DEVICE}")

    return model


def translate(
    text,
    model,
    src_tokenizer,
    tgt_tokenizer,
    max_length=Config.MAX_SEQ_LEN
):
    src_ids = src_tokenizer.encode(text)

    if len(src_ids) > max_length:
        src_ids = src_ids[:max_length - 1] + [Config.EOS_ID]

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

            generated.append(next_token)

            if next_token == Config.PAD_ID:
                break

    return tgt_tokenizer.decode(generated[1:])


def main():
    src_tokenizer = Tokenizer("tokenizers/en.model")
    tgt_tokenizer = Tokenizer("tokenizers/ru.model")

    model = load_model()

    while True:
        text = input("\nEnglish (or 'exit'): ").strip()

        if text.lower() == "exit":
            break

        if not text:
            continue

        result = translate(
            text,
            model,
            src_tokenizer,
            tgt_tokenizer
        )

        print("Russian:", result)


if __name__ == "__main__":
    main()
