import torch

from src.model.transformer_decoder import TransformerDecoder


def test_transformer_decoder():

    tgt = torch.randint(0, 16000, (2, 15))
    encoder_output = torch.randn(2, 20, 256)

    model = TransformerDecoder(
        vocab_size=16000,
        num_layers=4,
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(tgt, encoder_output)

    assert output.shape == (2, 15, 256)
    assert torch.isfinite(output).all()