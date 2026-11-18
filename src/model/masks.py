import torch


def generate_causal_mask(seq_len: int, device=None):

    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )

    return mask

def generate_padding_mask(sequence, padding_idx=0):

    mask = (sequence == padding_idx)

    return mask.unsqueeze(1).unsqueeze(2)

def generate_decoder_mask(sequence, padding_idx=0):

    seq_len = sequence.size(1)
    device = sequence.device

    causal_mask = generate_causal_mask(
        seq_len=seq_len,
        device=device
    )

    padding_mask = generate_padding_mask(
        sequence,
        padding_idx=padding_idx
    )

    return causal_mask.unsqueeze(0).unsqueeze(0) | padding_mask