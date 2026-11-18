import torch.nn as nn

from src.model.embeddings import TokenEmbedding
from src.model.positional_encoding import PositionalEncoding
from src.model.encoder import Encoder


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx
        )

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(self, src, src_padding_mask=None):

        x = self.embedding(src)
        x = self.positional_encoding(x)

        return self.encoder(
            x,
            padding_mask=src_padding_mask
        )