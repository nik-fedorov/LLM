from sentencepiece import SentencePieceProcessor


class SentencePieceTokenizer:
    def __init__(self, model_file):
        self.sp_model = SentencePieceProcessor(model_file=model_file)

    def encode(self, texts):
        return self.sp_model.encode(texts)

    def decode(self, texts):
        return self.sp_model.decode(texts)

    @property
    def pad_id(self):
        return self.sp_model.pad_id()

    @property
    def unk_id(self):
        return self.sp_model.unk_id()

    @property
    def bos_id(self):
        return self.sp_model.bos_id()

    @property
    def eos_id(self):
        return self.sp_model.eos_id()

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()
