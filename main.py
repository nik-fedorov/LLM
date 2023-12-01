import argparse
import logging

import torch
from torch.utils.data import DataLoader

from data import TinyStories, collate_fn
import loss
import models
from tokenizer import SentencePieceTokenizer
from train import train
from utils import init_obj, load_json


def main(config):
    logging.basicConfig(level=logging.INFO)

    tokenizer = SentencePieceTokenizer(**config['tokenizer'])

    train_set = TinyStories(**config['dataset'], train=True)
    test_set = TinyStories(**config['dataset'], train=False)
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, drop_last=True, **config['dataloader'])
    test_loader = DataLoader(test_set, shuffle=False, collate_fn=collate_fn, drop_last=True, **config['dataloader'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_obj(config['model'], models,
                     vocab_size=tokenizer.vocab_size,
                     pad_id=tokenizer.pad_id,
                     max_len=config['dataset']['max_len']).to(device)
    optimizer = init_obj(config['optimizer'], torch.optim, model.parameters())
    criterion = init_obj(config['criterion'], loss, ignore_index=tokenizer.pad_id)
    scheduler = init_obj(config['scheduler'], torch.optim.lr_scheduler, optimizer) if 'scheduler' in config else None

    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        config=config,
        device=device
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="train args")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args = args.parse_args()

    config = load_json(args.config)
    main(config)
