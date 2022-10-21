from dataclasses import dataclass, asdict
import os


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack: tuple = ()

    ### Training Parameters ###
    batch_size: int = 64
    lr: float = 0.1
    epochs: int = 40

    ### Model Parameters ###

    ## ResBlock Params ##
    block_depth: tuple = (2, 3, 2)
    block_channels: tuple = (64, 128, 256)
    kernel_size: int = 7  # Must be odd
    density: int = 4

    ### Sys Parameters ###
    data_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"  # Ubuntu
    # data_dir: os.PathLike = "/Users/josephbajor/Dev/Datasets/11-785-f22-hw2p2/"  # MacOS
    keyring_dir: os.PathLike = "/home/jbajor/Dev/keyring/"  # Ubuntu
    # keyring_dir: os.PathLike = "/Users/josephbajor/Dev/keyring/"  # MacOS

    ### WandB Parameters ###
    architecture: str = (
        f"ResNext-BN_mini_{sum(block_depth)}blocks_MaxC{max(block_channels)}"
    )
    project: str = "hw2p2-ablations"
    use_wandb: bool = True

    def wandb_export(self):
        to_exclude = ["data_dir", "keyring_dir", "use_wandb"]

        config = asdict(self)

        for param in to_exclude:
            del config[param]

        return config
