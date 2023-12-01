import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import wandb

from inference import generate
from utils import move_batch_to_device, save_model, load_model, inf_loop


logger = logging.getLogger(__name__)


PROMPTS = {
    '1': 'Once upon a time',
    '2': 'Alice was so tired when she got back home so she went',
    '3': 'Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked',
    '4': 'Once a little girl Lily went outside and saw',
}


@torch.no_grad()
def _validation_epoch(model, criterion, loader, config, device):
    model.eval()

    val_loss = 0.0
    for batch in tqdm(loader):
        move_batch_to_device(batch, device)

        output = model(**batch)
        batch.update(output)
        loss = criterion(**batch)
        val_loss += loss.item() * config['dataloader']['batch_size']

    return val_loss / len(loader.dataset)


def _train_epoch(model, optimizer, criterion, scheduler, train_loader, config, device,
                 len_epoch, log_period, epoch):
    model.train()

    train_loss = 0.0
    for batch_idx, batch in enumerate(
            tqdm(train_loader, desc="train", total=len_epoch)
    ):
        move_batch_to_device(batch, device)

        optimizer.zero_grad()
        output = model(**batch)
        batch.update(output)
        loss = criterion(**batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config["trainer"]["clip_grad_norm"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item() * config['dataloader']['batch_size']

        if (batch_idx + 1) % log_period == 0:
            train_loss /= config['dataloader']['batch_size'] * log_period
            wandb.log(
                {"loss/train": train_loss, 'perplexity/train': math.exp(train_loss)},
                step=epoch * len_epoch + batch_idx
            )
            train_loss = 0.0

        if batch_idx + 1 == len_epoch:
            break


def train(model, optimizer, criterion, scheduler, train_loader, test_loader, tokenizer, config, device):

    if config['trainer']['len_epoch'] is not None:
        # iteration based training
        len_epoch = config['trainer']['len_epoch']
        train_loader = inf_loop(train_loader)
    else:
        # epoch based training
        len_epoch = len(train_loader)

    # prepare directory for checkpoint saving
    save_dir = Path(config["trainer"]["checkpoint_save_dir"]) / config["trainer"]["wandb_run_name"]
    assert not save_dir.exists(), 'Choose new config["trainer"]["wandb_run_name"]'
    save_dir.mkdir(parents=True)

    # check for resuming training
    resume_config = config['trainer']['resume_training']
    if resume_config is not None:
        wandb_init_data = {
            'project': config["trainer"]["wandb_project"],
            'id': resume_config['wandb_run_id_to_resume'],
            'resume': 'must'
        }
        start_epoch = load_model(resume_config['checkpoint_path'], device, model, optimizer, scheduler)
    else:
        wandb_init_data = {
            'project': config["trainer"]["wandb_project"],
            'name': config["trainer"]["wandb_run_name"],
            'config': config
        }
        start_epoch = 0

    # count number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    with wandb.init(**wandb_init_data):
        logger.info(model)
        logger.info("Trainable parameters: {}".format(params))
        logger.info(f'Starting training from epoch {start_epoch}')
        logger.info('Training started')
        for epoch in range(start_epoch, start_epoch + config["trainer"]["num_epochs"]):
            logger.info(f'train epoch {epoch + 1} started: {dt.datetime.now()}')

            _train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                train_loader=train_loader,
                config=config,
                device=device,
                len_epoch=len_epoch,
                log_period=config['trainer']['log_period'],
                epoch=epoch
            )

            logger.info(f'validation after {epoch + 1} epochs started: {dt.datetime.now()}')

            valid_loss = _validation_epoch(
                model=model,
                criterion=criterion,
                loader=test_loader,
                config=config,
                device=device
            )

            logger.info(f'validation after {epoch + 1} epochs finished: {dt.datetime.now()}')

            wandb.log(
                {
                    "loss/val": valid_loss,
                    'perplexity/val': math.exp(valid_loss),
                    **{f'generated_text_{key}':  wandb.Html(
                        generate(model, prompt, tokenizer, config['dataset']['max_len'], device)
                       )
                       for key, prompt in PROMPTS.items()}
                },
                step=(epoch + 1) * len_epoch
            )

            if (epoch + 1) % config["trainer"]["checkpoint_save_period"] == 0:
                logger.info(f'Saving checkpoint after {epoch + 1} training epochs')
                ckpt_path = str(save_dir / f'checkpoint-epoch{epoch + 1}.pth')
                save_model(ckpt_path, epoch + 1, model, optimizer)
                if config['trainer']['enable_saving_checkpoints_in_wandb']:
                    wandb.save(ckpt_path)
