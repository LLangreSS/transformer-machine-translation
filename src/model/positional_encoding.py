import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(
            0, max_seq_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)