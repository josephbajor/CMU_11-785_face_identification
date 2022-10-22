from dataclasses import dataclass, asdict
import os
from torchvision import transforms


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack: tuple = (
        transforms.ColorJitter(
            brightness=(0.6, 1.4),
            saturation=(0.6, 1.4),
        ),
        transforms.RandomRotation(degrees=20, fill=0),
    )

    transform_stack: tuple = ()

    ### Training Parameters ###
    batch_size: int = 64
    lr: float = 0.1
    epochs: int = 40

    ### Model Parameters ###

    ## ResBlock Params ##
    block_depth: tuple = (3, 3, 9, 3)
    block_channels: tuple = (96, 192, 384, 768)
    kernel_size: int = 7  # Must be odd
    density: int = 4

    ### Sys Parameters ###
    data_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"  # Ubuntu
    # data_dir: os.PathLike = "/Users/josephbajor/Dev/Datasets/11-785-f22-hw2p2/"  # MacOS
    keyring_dir: os.PathLike = "/home/jbajor/Dev/keyring/"  # Ubuntu
    # keyring_dir: os.PathLike = "/Users/josephbajor/Dev/keyring/"  # MacOS
    model_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/models/"

    ### WandB Parameters ###
    architecture: str = (
        f"ResNext-BN_XL_v3_{sum(block_depth)}blocks_MaxC{max(block_channels)}"
    )
    project: str = "hw2p2-ablations"
    use_wandb: bool = True

    def wandb_export(self):
        to_exclude = ["data_dir", "keyring_dir", "use_wandb"]

        config = asdict(self)

        for param in to_exclude:
            del config[param]

        return config
