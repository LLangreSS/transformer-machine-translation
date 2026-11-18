import sentencepiece as spm


class Tokenizer:

    def __init__(self, model_path):
        self.processor = spm.SentencePieceProcessor(
            model_file=str(model_path)
        )

        self.pad_id = self.processor.pad_id()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.unk_id = self.processor.unk_id()

    def encode(self, text):

        return self.processor.encode(
            text,
            out_type=int,
            add_bos=True,
            add_eos=True
        )

    def decode(self, ids):

        return self.processor.decode(ids)

    @property
    def vocab_size(self):
        return self.processor.get_piece_size()