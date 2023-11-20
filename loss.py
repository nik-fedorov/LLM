import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, text_encoded, logits, **batch):
        return self.loss(logits[:, :-1].transpose(1, 2), text_encoded[:, 1:])
