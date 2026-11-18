import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            "d_model must be divisible by num_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K and V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):

        # 1. Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        # True in mask means that position is blocked
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # 4. Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. Weighted sum of Values
        context = torch.matmul(attention_weights, V)

        # 6. Concatenate heads
        context = context.transpose(1, 2).contiguous()

        batch_size, seq_len, _, _ = context.shape

        context = context.view(
            batch_size,
            seq_len,
            self.d_model
        )

        # 7. Final projection
        output = self.W_o(context)

        return output
