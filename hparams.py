from dataclasses import dataclass, asdict
import os


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack: tuple = ()

    ### Training Parameters ###
    batch_size: int = 128
    lr: float = 0.1
    epochs: int = 25

    ### Model Parameters ###

    ## ResBlock Params ##
    block_depth: tuple = (3, 3, 9, 3)
    block_channels: tuple = (96, 192, 384, 768)
    kernel_size: int = 3
    padding: int = 3
    density: int = 4

    ### Sys Parameters ###
    data_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"
    keyring_dir: os.PathLike = "/home/jbajor/Dev/keyring/"

    ### WandB Parameters ###
    architecture: str = ""

    def wandb_export(self):
        to_exclude = ["data_dir", "keyring_dir"]

        config = asdict(self)

        for param in to_exclude:
            del config[param]

        return config
