import torch

from src.model.encoder import EncoderBlock, Encoder


def test_encoder_block():

    x = torch.randn(2, 10, 256)

    model = EncoderBlock(
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(x)

    assert output.shape == (2, 10, 256)
    assert torch.isfinite(output).all()

    print("EncoderBlock test passed!")


def test_encoder():

    x = torch.randn(2, 10, 256)

    model = Encoder(
        num_layers=4,
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(x)

    assert output.shape == (2, 10, 256)
    assert torch.isfinite(output).all()

    print("Encoder test passed!")


def test_gradients():

    x = torch.randn(2, 10, 256, requires_grad=True)

    model = Encoder(
        num_layers=2,
        d_model=256,
        num_heads=4,
        d_ff=1024
    )

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("Gradient test passed!")


if __name__ == "__main__":
    test_encoder_block()
    test_encoder()
    test_gradients()