from dataclasses import dataclass, asdict
import os
from torchvision import transforms


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack_PIL: tuple = (
        transforms.ColorJitter(
            brightness=(0.7, 1.4), saturation=(0.7, 1.4), hue=(0.065)
        ),
        transforms.RandomAdjustSharpness(0.2),
        transforms.RandomPerspective(0.5, p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
    )

    transform_stack_tensor: tuple = (transforms.RandomErasing(p=0.3),)

    ### Training Parameters ###
    batch_size: int = 128
    lr: float = 0.1
    epochs: int = 40

    ### Model Parameters ###

    ## ResBlock Params ##
    block_depth: tuple = (3, 3, 9, 3)
    block_channels: tuple = (96, 192, 384, 768)
    kernel_size: int = 7  # Must be odd
    density: int = 4

    ### Sys Parameters ###

    platform: str = "mac"

    if platform == "desktop":
        data_dir: os.PathLike = (
            "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"  # Ubuntu Local
        )
        keyring_dir: os.PathLike = "/home/jbajor/Dev/keyring/"  # Ubuntu Local
        model_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/models/"  # Ubuntu Local

    if platform == "mac":
        data_dir: os.PathLike = (
            "/Users/josephbajor/Dev/Datasets/11-785-f22-hw2p2/"  # MacOS
        )
        keyring_dir: os.PathLike = "/Users/josephbajor/Dev/keyring/"  # MacOS
        model_dir: os.PathLike = "/Users/josephbajor/Dev/CMU-IDL/models"  # MacOS

    if platform == "cloud":
        data_dir: os.PathLike = "/home/josephbajor/data/"  # CompEng
        keyring_dir: os.PathLike = "/home/josephbajor/keyring/"  # CompEng
        model_dir: os.PathLike = "/home/josephbajor/models/"  # CompEng

    ### WandB Parameters ###
    architecture: str = (
        f"ResNext-BN_Tform-LS_v4_{sum(block_depth)}blocks_MaxC{max(block_channels)}"
    )
    project: str = "hw2p2-ablations"
    use_wandb: bool = True

    def wandb_export(self):
        to_exclude = [
            "data_dir",
            "keyring_dir",
            "model_dir",
            "use_wandb",
        ]

        config = asdict(self)

        for param in to_exclude:
            del config[param]

        return config
