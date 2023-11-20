from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class TinyStories(Dataset):
    def __init__(self, dataset_dir: str, max_len: int, train: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.max_len = max_len

        split = 'train' if train else 'test'
        self.indices = np.load(str(self.dataset_dir / f'{split}_indices.npy'))

    def __getitem__(self, item):
        ind = self.indices[item]
        encoded = np.load(str(self.dataset_dir / f'{ind}.npy'))
        encoded = encoded[:self.max_len]
        return {'text_encoded': torch.from_numpy(encoded)}

    def __len__(self):
        return len(self.indices)


def collate_fn(dataset_items):
    text_encoded_length = torch.tensor([len(item['text_encoded']) for item in dataset_items])
    max_text_len = max(text_encoded_length)
    text_encoded = torch.full((len(dataset_items), max_text_len), 0, dtype=torch.long)
    for i, item in enumerate(dataset_items):
        text_encoded[i, :text_encoded_length[i]] = item['text_encoded']

    return {'text_encoded': text_encoded, 'lengths': text_encoded_length, 'batch_size': len(dataset_items)}
