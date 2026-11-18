import torch
import torch.nn as nn

from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward


class DecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Masked Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-Attention
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed Forward
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        causal_mask=None,
        src_padding_mask=None
    ):

        # 1. Masked Self-Attention
        self_attn_output = self.self_attention(
            x, x, x,
            mask=causal_mask
        )

        x = self.norm1(
            x + self.dropout(self_attn_output)
        )

        # 2. Cross-Attention
        cross_attn_output = self.cross_attention(
            x,
            encoder_output,
            encoder_output,
            mask=src_padding_mask
        )

        x = self.norm2(
            x + self.dropout(cross_attn_output)
        )

        # 3. Feed Forward
        ff_output = self.feed_forward(x)

        x = self.norm3(
            x + self.dropout(ff_output)
        )

        return x
    
class Decoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        encoder_output,
        causal_mask=None,
        src_padding_mask=None
    ):

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                causal_mask=causal_mask,
                src_padding_mask=src_padding_mask
            )

        return x
