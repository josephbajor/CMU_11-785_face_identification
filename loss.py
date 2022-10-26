import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    """Center Loss
        Center Loss Paper:
        https://ydwen.github.io/papers/WenECCV16.pdf
    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_classes=7000, feat_dim=1024):
        NotImplementedError
