from hparams import Hparams
import time
import os
import wandb
import torch
import torchvision
from random import randrange


def initiate_run(hparams: Hparams):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    with open(hparams.keyring_dir + "wandb.key") as key:
        wandb.login(key=key.read().strip())
        key.close()

    if hparams.use_wandb:
        mode = "online"
    else:
        mode = "disabled"

    run = wandb.init(
        name=f"{hparams.architecture}_{int(time.time())}",
        project=hparams.project,
        config=hparams.wandb_export(),
        mode=mode,
    )

    return run


def load_model(hparams, model, optimizer, scheduler=None):

    model_pth = os.path.join(
        hparams.model_dir, f"{hparams.architecture}/checkpoint.pth"
    )

    params = torch.load(model_pth)
    model.load_state_dict(params["model_state_dict"])
    optimizer.load_state_dict(params["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(params["scheduler_state_dict"])

        return model, optimizer, scheduler

    return model, optimizer


def prepare_instance() -> None:
    return NotImplementedError


def test_transforms(hparams, num: int = 5, start: int = -1) -> None:
    TRAIN_DIR = os.path.join(hparams.data_dir, "classification/train")
    transform_stack = list(hparams.transform_stack)
    train_transforms = torchvision.transforms.Compose(transform_stack)
    train_dataset = torchvision.datasets.ImageFolder(
        TRAIN_DIR, transform=train_transforms
    )

    if start == -1:
        offset = randrange(start=0, stop=len(train_dataset) - num)
    else:
        offset = start

    for i in range(offset, offset + num):
        display(train_dataset[i][0])
