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


class ArcFace(nn.Module):
    """
    Implementation of the ArcFace metric
    """

    def __init__(self, in_feats, out_feats, device, s=64.0, m=0.5):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.s = s
        self.m = m

        self.eps = 1e-7

        # Create weights for the FC layer
        self.W = nn.Parameter(torch.FloatTensor(self.out_feats, self.in_feats))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x, y):

        theta = F.linear(F.normalize(x), F.normalize(self.W))

        numerator = self.s * torch.cos(
            torch.acos(
                torch.clamp(
                    torch.diagonal(theta.transpose(0, 1)[y]),
                    -1.0 + self.eps,
                    1 - self.eps,
                )
            )
            + self.m
        )

        excl = torch.cat(
            [
                torch.cat((theta[i, :y], theta[i, y + 1 :])).unsqueeze(0)
                for i, y in enumerate(y)
            ],
            dim=0,
        )

        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        return -torch.mean(numerator - torch.log(denominator))
