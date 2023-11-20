import json
import os
from pathlib import Path
import random

import numpy as np
from sentencepiece import SentencePieceTrainer
from tqdm import tqdm

from tokenizer import SentencePieceTokenizer


DATASET_DIR = Path('TinyStories')
N_DATA_FILES = 50

# SentencePieceTrainer params
VOCAB_SIZE = 3000
MODEL_TYPE = 'bpe'
MODEL_PREFIX = 'spm'
NORMALIZATION_RULE_NAME = 'nmt_nfkc'
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


random.seed(42)


def prepare_spm_model():
    # create train corpus for sentencepiece model
    filename = 'spm_train_texts.txt'
    with open(filename, 'w') as train_file:
        for i in random.sample(range(N_DATA_FILES), 10):
            with open(DATASET_DIR / ('data%02d.json' % i)) as f:
                data = json.load(f)
                for item in data:
                    train_file.write(item['story'] + '\n')

    # train tokenizer
    SentencePieceTrainer.train(
        input=filename, vocab_size=VOCAB_SIZE, model_type=MODEL_TYPE, model_prefix=MODEL_PREFIX,
        normalization_rule_name=NORMALIZATION_RULE_NAME,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID
    )
    os.remove(filename)


def prepare_tiny_stories_dataset():
    # prepare encoded texts
    tokenizer = SentencePieceTokenizer(MODEL_PREFIX + '.model')
    counter = 0
    for i in tqdm(range(N_DATA_FILES), desc='Encoding train texts'):
        with open(DATASET_DIR / ('data%02d.json' % i)) as f:
            data = json.load(f)
        texts = [item['story'] for item in data]
        encoded_texts = tokenizer.encode(texts)
        for encoded in encoded_texts:
            # encoded = tokenizer.encode(item['story'])
            encoded = [tokenizer.bos_id] + encoded + [tokenizer.eos_id]
            np.save(str(DATASET_DIR / f'{counter}.npy'), np.array(encoded))
            counter += 1

    # prepare train/test split
    # counter = 4967871
    counter = 100000
    indices = list(range(counter))
    random.shuffle(indices)
    train_len = int(0.95 * counter)
    np.save(str(DATASET_DIR / 'train_indices.npy'), np.array(indices[:train_len]))
    np.save(str(DATASET_DIR / 'test_indices.npy'), np.array(indices[train_len:]))


prepare_spm_model()
prepare_tiny_stories_dataset()
