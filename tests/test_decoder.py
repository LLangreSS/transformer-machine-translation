import torch

from src.model.decoder import DecoderBlock, Decoder


def test_decoder_block():

    x = torch.randn(2, 15, 256)
    encoder_output = torch.randn(2, 20, 256)

    model = DecoderBlock(
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(x, encoder_output)

    assert output.shape == (2, 15, 256)
    assert torch.isfinite(output).all()


def test_decoder():

    x = torch.randn(2, 15, 256)
    encoder_output = torch.randn(2, 20, 256)

    model = Decoder(
        num_layers=4,
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(x, encoder_output)

    assert output.shape == (2, 15, 256)
    assert torch.isfinite(output).all()