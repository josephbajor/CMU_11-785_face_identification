from dataclasses import dataclass, asdict
import os


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack: tuple = ()

    ### Training Parameters ###
    batch_size: int = 64
    lr: float = 0.1
    epochs: int = 25

    ### Model Parameters ###

    ## ResBlock Params ##
    block_depth: tuple = (3, 3, 9, 3)
    block_channels: tuple = (96, 192, 384, 768)
    kernel_size: int = 3  # Must be odd
    density: int = 4

    ### Sys Parameters ###
    data_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"
    keyring_dir: os.PathLike = "/home/jbajor/Dev/keyring/"

    ### WandB Parameters ###
    architecture: str = "ResEXP_v1_micro"
    project: str = "hw2p2-ablations"
    use_wandb: bool = False

    def wandb_export(self):
        to_exclude = ["data_dir", "keyring_dir", "use_wandb"]

        config = asdict(self)

        for param in to_exclude:
            del config[param]

        return config
