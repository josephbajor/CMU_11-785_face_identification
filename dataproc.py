from hparams import Hparams
import os
import torchvision
import torch
from PIL import Image


class ClassificationTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths = list(
            map(
                lambda fname: os.path.join(self.data_dir, fname),
                sorted(os.listdir(self.data_dir)),
            )
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


def build_loaders(hparams: Hparams):

    TRAIN_DIR = os.path.join(hparams.data_dir, "classification/train")
    VAL_DIR = os.path.join(hparams.data_dir, "classification/dev")
    TEST_DIR = os.path.join(hparams.data_dir, "classification/test")

    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(
        TRAIN_DIR, transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    test_dataset = ClassificationTestDataset(TEST_DIR, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hparams.batch_size, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader
