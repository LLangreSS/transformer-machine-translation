
import torch
from src.model.attention import MultiHeadAttention


def test_attention():

    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 4

    attention = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads
    )

    x = torch.randn(batch_size, seq_len, d_model)

    output = attention(x, x, x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 10, 256)

    print("Attention test passed!")


if __name__ == "__main__":
    test_attention()
