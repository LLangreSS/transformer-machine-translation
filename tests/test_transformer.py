import torch

from src.model.transformer import Transformer


def test_transformer():

    src = torch.randint(1, 16000, (2, 20))
    tgt = torch.randint(1, 16000, (2, 15))

    # Add padding
    src[:, -2:] = 0
    tgt[:, -3:] = 0

    model = Transformer(
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        d_model=256,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024
    )

    logits = model(src, tgt)

    assert logits.shape == (2, 15, 16000)
    assert torch.isfinite(logits).all()

def test_transformer_backward():

    src = torch.randint(1, 16000, (2, 20))
    tgt = torch.randint(1, 16000, (2, 15))

    model = Transformer(
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        d_model=256,
        num_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_ff=1024
    )

    logits = model(src, tgt)

    loss = logits.mean()
    loss.backward()

    assert model.encoder.embedding.embedding.weight.grad is not None
    assert model.decoder.embedding.embedding.weight.grad is not None
    assert model.output_projection.weight.grad is not None