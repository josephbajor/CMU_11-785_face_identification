import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams


class LoPassNetwork(torch.nn.Module):
    """
    The Very Low early deadline architecture is a 4-layer CNN.

    The first Conv layer has 64 channels, kernel size 7, and stride 4.
    The next three have 128, 256, and 512 channels. Each have kernel size 3 and stride 2.

    Think about strided convolutions from the lecture, as convolutioin with stride= 1 and downsampling.
    For stride 1 convolution, what padding do you need for preserving the spatial resolution?
    (Hint => padding = kernel_size // 2) - Why?)

    Each Conv layer is accompanied by a Batchnorm and ReLU layer.
    Finally, you want to average pool over the spatial dimensions to reduce them to 1 x 1. Use AdaptiveAvgPool2d.
    Then, remove (Flatten?) these trivial 1x1 dimensions away.
    Look through https://pytorch.org/docs/stable/nn.html

    TODO: Fill out the model definition below!

    Why does a very simple network have 4 convolutions?
    Input images are 224x224. Note that each of these convolutions downsample.
    Downsampling 2x effectively doubles the receptive field, increasing the spatial
    region each pixel extracts features from. Downsampling 32x is standard
    for most image models.

    Why does a very simple network have high channel sizes?
    Every time you downsample 2x, you do 4x less computation (at same channel size).
    To maintain the same level of computation, you 2x increase # of channels, which
    increases computation by 4x. So, balances out to same computation.
    Another intuition is - as you downsample, you lose spatial information. We want
    to preserve some of it in the channel dimension.
    """

    def __init__(self, num_classes=7000):
        super().__init__()

        self.backbone = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=4, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
        )

        self.cls_layer = nn.Linear(512, num_classes)

    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        out = self.cls_layer(feats.flatten(start_dim=1))

        if return_feats:
            return feats
        else:
            return out

    def debug(self, input):
        out = input
        for m in self.backbone.children():
            out = m(out)
            print(m, out.shape)
        return out


# Within each ConvNext block, the input dims will always equal the output dims
class ConvNextBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, density) -> None:
        super().__init__()
        # self.channles = channels
        # self.kernel_size = kernel_size
        # self.padding = padding

        self.l1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.lnorm = nn.LayerNorm(channels, eps=1e-6)
        self.l2 = nn.Conv2d(channels, channels * density, kernel_size=1)
        # GELU activation between l2 and l3
        self.l3 = nn.Conv2d(channels * density, channels, kernel_size=1)

    def forward(self, x):
        out = self.l1(x)
        out = self.lnorm(out)
        out = self.l2(out)
        out = self.l3(F.gelu(out))

        return out + x


class ResEXP(nn.Module):
    def __init__(self, hparams: Hparams, channels=3, classes=7000) -> None:
        super().__init__()
        self.hparams = hparams

        self.network = nn.ModuleList()

        # initial layer
        self.network.extend(
            [
                nn.Conv2d(
                    channels, self.hparams.block_channels[0], kernel_size=4, stride=4
                ),
                nn.LayerNorm(self.hparams.block_channels[0], eps=1e-6),
            ]
        )

        for i in range(len(self.hparams.block_channels)):
            layers = [
                ConvNextBlock(
                    self.hparams.block_channels[i],
                    kernel_size=hparams.kernel_size,
                    padding=hparams.padding,
                    density=hparams.density,
                )
                for block in range(hparams.block_depth[i])
            ]
            if i != len(self.hparams.block_channels) - 1:
                # Append downsampling layer
                layers.extend(
                    [
                        nn.LayerNorm(self.hparams.block_channels[i], eps=1e-6),
                        nn.Conv2d(
                            self.hparams.block_channels[i],
                            self.hparams.block_channels[i + 1],
                            kernel_size=2,
                            stride=2,
                        ),
                    ]
                )
            else:
                layers.append(nn.LayerNorm(self.hparams.block_channels[i], eps=1e-6))

            self.network.extend(layers)

        # Model head
        self.head = nn.Linear(self.hparams.block_channels[-1], classes)

    def forward(self, x, headless=False):
        for layer in self.network:
            x = layer(x)
        if headless:
            return x
        return self.head(x)
