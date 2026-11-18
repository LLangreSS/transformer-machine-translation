import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int = 0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )

        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)