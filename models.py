import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * \
                          (-math.log(10000) / embed_dim)).unsqueeze(0)

        arguments = positions * freqs
        pe[:, 0::2] = torch.sin(arguments)
        pe[:, 1::2] = torch.cos(arguments)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class NikitosGPT(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, max_len, vocab_size, pad_id):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, text_encoded, lengths, **batch):
        x = self.embed(text_encoded) * math.sqrt(self.d_model)  # B x L x E
        x = self.pos_enc(x)
        x = self.encoder(x,
                         self.generate_square_subsequent_mask(text_encoded.size(1)).to(x.device),
                         self.get_pad_mask_from_lengths(lengths).to(x.device))
        x = self.fc(x)
        return {'logits': x}

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def get_pad_mask_from_lengths(lengths, max_len=None):
        if max_len == None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len, 1, device=lengths.device)
        mask = (ids >= lengths.unsqueeze(1)).bool()

        return mask
