if __name__ == "__main__":
    from hparams import Hparams
    from model import ResEXP
    from dataproc import build_loaders
    import torch
    import torch.nn as nn
    import os
    import torchvision

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    hparams = Hparams()

    DATA_DIR = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"

    TRAIN_DIR = os.path.join(DATA_DIR, "classification/train")

    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        TRAIN_DIR, transform=train_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = ResEXP(hparams).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams.lr)
    scaler = torch.cuda.amp.GradScaler()

    num_correct = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):

        print(i)
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
