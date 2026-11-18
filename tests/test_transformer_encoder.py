import torch

from src.model.transformer_encoder import TransformerEncoder


def test_transformer_encoder():

    model = TransformerEncoder(
        vocab_size=16000,
        d_model=256,
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_seq_len=256
    )

    # Batch: 32 предложения по 20 токенов
    src = torch.randint(1, 16000, (32, 20))

    output = model(src)

    assert output.shape == (32, 20, 256)
    assert torch.isfinite(output).all()

    print("TransformerEncoder test passed!")


if __name__ == "__main__":
    test_transformer_encoder()