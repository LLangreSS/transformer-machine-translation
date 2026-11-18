import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):

    def __init__(
        self,
        src_sequences,
        tgt_sequences
    ):
        assert len(src_sequences) == len(tgt_sequences)

        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):

        src = torch.tensor(
            self.src_sequences[idx],
            dtype=torch.long
        )

        tgt = torch.tensor(
            self.tgt_sequences[idx],
            dtype=torch.long
        )

        return src, tgt

def collate_fn(batch, padding_idx=0):

    src_sequences, tgt_sequences = zip(*batch)

    src = pad_sequence(
        src_sequences,
        batch_first=True,
        padding_value=padding_idx
    )

    tgt = pad_sequence(
        tgt_sequences,
        batch_first=True,
        padding_value=padding_idx
    )

    return src, tgt