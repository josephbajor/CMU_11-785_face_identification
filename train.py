import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataproc import build_loaders
from hparams import Hparams
from model import ResNext_BN, ResNext
from utils import initiate_run
import wandb


def train(model, hparams, dataloader, optimizer, criterion, scaler, device):

    model.train()

    # Progress Bar
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=5,
    )

    num_correct = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (hparams.batch_size * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        # TODO? Depending on your choice of scheduler,
        # You may want to call some schdulers inside the train function. What are these?

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (hparams.batch_size * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, hparams, dataloader, criterion, device):

    model.eval()
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val",
        ncols=5,
    )

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (hparams.batch_size * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
        )

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (hparams.batch_size * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss


def main(
    hparams: Hparams, device_override: str = None, warm_start: bool = False
) -> None:

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    if device_override:
        device = device_override

    model_pth = os.path.join(
        hparams.model_dir, f"{hparams.architecture}/checkpoint.pth"
    )
    os.makedirs(
        os.path.join(hparams.model_dir, f"{hparams.architecture}/"), exist_ok=True
    )

    train_loader, val_loader, test_loader = build_loaders(hparams)

    if hparams.model == "ResNext":
        model = ResNext(hparams).to(device)
    if hparams.model == "ResNext-BN":
        model = ResNext_BN(hparams).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    if hparams.optim_func == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
        )
    if hparams.optim_func == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay
        )
    else:
        assert NameError, "optim_func must be AdamW or SGD!"

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, patience=3
    )
    scaler = torch.cuda.amp.GradScaler()

    epoch_offset = 0

    if warm_start:
        params = torch.load(model_pth)
        model.load_state_dict(params["model_state_dict"])
        optimizer.load_state_dict(params["optimizer_state_dict"])
        epoch_offset = params["epoch"]

    run = initiate_run(hparams)

    wandb.watch(model, log="all")

    best_valacc = 0.0

    wandb.config.update(
        {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    )

    for epoch in range(epoch_offset, hparams.epochs):

        curr_lr = float(optimizer.param_groups[0]["lr"])

        train_acc, train_loss = train(
            model, hparams, train_loader, optimizer, criterion, scaler, device
        )

        print(
            "\nEpoch {}/{}: \nTrain Acc {:.04f}%\t Train Loss {:.04f}\t Learning Rate {:.04f}".format(
                epoch + 1, hparams.epochs, train_acc, train_loss, curr_lr
            )
        )

        val_acc, val_loss = validate(model, hparams, val_loader, criterion, device)

        print("Val Acc {:.04f}%\t Val Loss {:.04f}".format(val_acc, val_loss))

        scheduler.step(epoch)

        wandb.log(
            {
                "train_loss": train_loss,
                "train_Acc": train_acc,
                "validation_Acc": val_acc,
                "validation_loss": val_loss,
                "learning_Rate": scheduler.get_last_lr(),
            }
        )

        # #Save model in drive location if val_acc is better than best recorded val_acc
        if val_acc >= best_valacc:
            # path = os.path.join(root, model_directory, 'checkpoint' + '.pth')
            print("Saving model...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                model_pth,
            )
            print(f"Saved to {model_pth}")
            best_valacc = val_acc
            if val_acc > 0.8:
                wandb.save(model_pth)
            # You may find it interesting to exlplore Wandb Artifcats to version your models
    run.finish()


if __name__ == "__main__":
    hparams = Hparams()
    main(hparams, device_override=None)
