import datetime as dt
import logging
import math

import numpy as np
import torch
from tqdm import tqdm
import wandb

from utils import move_batch_to_device, save_model


logger = logging.getLogger(__name__)


@torch.no_grad()
def _validation_epoch(model, criterion, loader, device):
    model.eval()

    loss = 0.0
    for batch in tqdm(loader):
        move_batch_to_device(batch, device)

        output = model(**batch)
        batch.update(output)
        loss = criterion(**batch)
        loss += loss.item() * batch['batch_size']

    n_samples = len(loader.dataset)
    return loss / n_samples, math.exp(loss / n_samples)


def _train_epoch(model, optimizer, criterion, train_loader, config, device):
    model.train()

    loss = 0.0
    for batch in tqdm(train_loader):
        move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(**batch)
        batch.update(output)
        loss = criterion(**batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["trainer"]["clip_grad_norm"])
        optimizer.step()

        loss += loss.item() * batch['batch_size']

    n_samples = len(train_loader.dataset)
    return loss / n_samples, math.exp(loss / n_samples)


def train(model, optimizer, criterion, train_loader, test_loader, config, device):
    wandb_init_data = {
        'project': config["trainer"]["wandb_project"],
        'config': config
    }

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Trainable parameters: {}".format(params))

    with wandb.init(**wandb_init_data):
        logger.info('Training started')
        for epoch in range(config["trainer"]["num_epochs"]):
            logger.info(f'train epoch {epoch + 1} started: {dt.datetime.now()}')

            train_loss, train_perplexity = _train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                config=config,
                device=device
            )

            logger.info(f'validation after {epoch + 1} epochs started: {dt.datetime.now()}')

            valid_loss, valid_perplexity = _validation_epoch(
                model=model,
                criterion=criterion,
                loader=test_loader,
                device=device
            )

            logger.info(f'validation after {epoch + 1} epochs finished: {dt.datetime.now()}')

            wandb.log({"loss/train": train_loss, "loss/val": valid_loss,
                       'perplexity/train': train_perplexity, 'perplexity/val': valid_perplexity})

            if (epoch + 1) % config["trainer"]["checkpoint_save_period"] == 0:
                logger.info(f'Saving checkpoint after {epoch + 1} training epochs')
                ckpt_name = f'checkpoint-epoch{epoch + 1}.pth'
                save_model(ckpt_name, epoch + 1, model, optimizer)
                wandb.save(ckpt_name)
