from dataclasses import dataclass
import os


@dataclass
class Hparams:

    ### Preprocessing Parameters ###
    transform_stack: list = []

    ### Training Parameters ###
    batch_size: int = 128
    lr: float = 0.1
    epochs: int = 25

    ### Model Parameters ###

    ### Sys Parameters ###
    data_dir: os.PathLike = "/home/jbajor/Dev/CMU-IDL/datasets/hw2p2/"

    def wandb_export(self):

        to_exclude = ["placeholder"]
