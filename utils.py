import importlib
from itertools import repeat
import json

import torch


# I took this function from DLA ASR homework
def init_obj(obj_dict, default_module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj(config['param'], module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    if "module" in obj_dict:
        default_module = importlib.import_module(obj_dict["module"])

    module_name = obj_dict["type"]
    module_args = dict(obj_dict["args"])
    assert all(
        [k not in module_args for k in kwargs]
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return getattr(default_module, module_name)(*args, **module_args)


def load_json(path):
    with open(path) as f:
        content = json.load(f)
    return content


def save_model(path, num_epochs, model, optimizer, scheduler=None):
    '''Save on GPU'''
    data = {
        'num_epochs': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    torch.save(data, path)


def load_model(path, device, model, optimizer=None, scheduler=None):
    '''Load on GPU'''
    data = torch.load(path)
    model.load_state_dict(data['model_state_dict'])
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(data['scheduler_state_dict'])
    return data['num_epochs']


def move_batch_to_device(data, device):
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader
