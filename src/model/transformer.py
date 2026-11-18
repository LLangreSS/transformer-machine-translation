import torch.nn as nn

from src.model.transformer_encoder import TransformerEncoder
from src.model.transformer_decoder import TransformerDecoder
from src.model.masks import (
    generate_padding_mask,
    generate_decoder_mask
)


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        src_padding_idx: int = 0,
        tgt_padding_idx: int = 0
    ):
        super().__init__()

        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=src_padding_idx
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=tgt_padding_idx
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def forward(self, src, tgt):

        # 1. Create masks
        src_padding_mask = generate_padding_mask(
            src,
            padding_idx=self.src_padding_idx
        )

        tgt_mask = generate_decoder_mask(
            tgt,
            padding_idx=self.tgt_padding_idx
        )

        # 2. Encoder
        encoder_output = self.encoder(
            src,
            src_padding_mask=src_padding_mask
        )

        # 3. Decoder
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            causal_mask=tgt_mask,
            src_padding_mask=src_padding_mask
        )

        # 4. Project to target vocabulary
        logits = self.output_projection(decoder_output)

        return logits