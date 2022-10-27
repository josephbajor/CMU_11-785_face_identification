import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import Hparams
from torchvision.ops import stochastic_depth
import random
import numpy as np


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


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, density) -> None:
        super().__init__()

        self.padding: int = (kernel_size - 1) // 2

        self.core = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_ch),
        )


class ConvNextBlock_BN(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        density: int,
        drop_block: bool = False,
        drop_path: bool = False,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()
        # shape of the input must remain the same as the output
        self.padding: int = (kernel_size - 1) // 2  # assumes stride == 1

        self.p_drop = p_drop
        self.drop_block = drop_block
        self.drop_path = drop_path

        self.conv_dwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
        )
        self.bnorm = nn.BatchNorm2d(channels)
        self.conv_pwise_1 = nn.Conv2d(channels, channels * density, kernel_size=1)
        self.conv_pwise_2 = nn.Conv2d(channels * density, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, training=False):
        residual = x

        if (
            self.p_drop != 0.0
            and self.drop_block
            and random.random() < (self.p_drop)
            and training
        ):
            return residual

        x = self.conv_dwise(x)
        x = self.bnorm(x)
        x = self.conv_pwise_1(F.gelu(x))
        x = self.conv_pwise_2(F.gelu(x))

        if self.p_drop != 0.0 and self.drop_path:
            return x + stochastic_depth(
                residual, p=self.p_drop, mode="batch", training=training
            )

        return x + residual


class ResNext_BN(nn.Module):
    def __init__(self, hparams: Hparams, in_channels=3, classes=7000) -> None:
        super().__init__()

        self.hparams = hparams

        self.backbone = nn.ModuleList()

        if self.hparams.drop_blocks or self.hparams.drop_path:
            self.drop_probs = np.linspace(
                start=0,
                stop=self.hparams.max_drop_prob,
                num=sum(self.hparams.block_depth),
            )

        # initial layers
        self.backbone.extend(
            [
                nn.Conv2d(
                    in_channels, self.hparams.block_channels[0], kernel_size=4, stride=4
                ),
                nn.BatchNorm2d(hparams.block_channels[0]),
            ]
        )
        ### Core Network Initialization ###

        for i in range(len(self.hparams.block_channels)):

            if i == 0:
                past = 0
            else:
                past = past + self.hparams.block_depth[i - 1]

            layers = [
                ConvNextBlock_BN(
                    self.hparams.block_channels[i],
                    kernel_size=hparams.kernel_size,
                    density=hparams.density,
                    drop_block=self.hparams.drop_blocks,
                    drop_path=self.hparams.drop_path,
                    p_drop=self.drop_probs[past + block]
                    if self.hparams.drop_blocks
                    else 0.0,
                )
                for block in range(hparams.block_depth[i])
            ]

            if i != len(self.hparams.block_channels) - 1:
                # Append downsampling components
                layers.extend(
                    [
                        nn.BatchNorm2d(self.hparams.block_channels[i]),
                        nn.Conv2d(
                            self.hparams.block_channels[i],
                            self.hparams.block_channels[i + 1],
                            kernel_size=2,
                            stride=2,
                        ),
                    ]
                )

            else:
                layers.extend(
                    [
                        nn.BatchNorm2d(self.hparams.block_channels[i]),
                        nn.AdaptiveAvgPool2d(output_size=1),
                    ]
                )

            self.backbone.extend(layers)
        ### END Core Network Initialization ###

        # Model head
        self.head = nn.Linear(self.hparams.block_channels[-1], classes)

    def forward(self, x: torch.Tensor, headless=False):

        for layer in self.backbone:
            if isinstance(layer, ConvNextBlock_BN):
                x = layer(x, self.training)
            else:
                x = layer(x)

        x = x.flatten(start_dim=1)

        if headless:
            return x, self.head(x)
        return self.head(x)


# Within each ConvNext block, the input dims will always equal the output dims
class ConvNextBlock(nn.Module):
    # TODO: Add dynamic layernorm setup

    def __init__(self, channels, kernel_size, density) -> None:
        super().__init__()

        # shape of the input must remain the same as the output
        self.padding: int = (kernel_size - 1) // 2  # assumes stride == 1

        self.l1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,
        )
        self.lnorm = nn.LayerNorm(channels, eps=1e-6)
        self.l2 = nn.Linear(channels, channels * density)
        # GELU activation between l2 and l3
        self.l3 = nn.Linear(channels * density, channels)

    def forward(self, x: torch.Tensor):
        # Assume that input comes with channels last (B,Y,X,C)
        residual = x
        x = x.permute(0, 3, 1, 2)  # (B,Y,X,C) -> (B,C,Y,X)
        x = self.l1(x)
        x = x.permute(0, 2, 3, 1)  # (B,C,Y,X) -> (B,Y,X,C)
        x = self.lnorm(x)
        x = self.l2(F.gelu(x))
        x = self.l3(F.gelu(x))

        return x + residual


class ResNext(nn.Module):
    def __init__(self, hparams: Hparams, in_channels=3, classes=7000) -> None:
        super().__init__()
        self.hparams = hparams

        # self.core_network = nn.ModuleList()

        # initial layer
        self.initial_layer = nn.Conv2d(
            in_channels, self.hparams.block_channels[0], kernel_size=4, stride=4
        )

        ### Core Network Initialization ###
        # NOTE: expects input as channels first (B,X,Y,C)

        # Due to how layernorm is implemented in pytorch, permutations are needed between
        # layernorm and conv layers. To deal with this, we generate a dictionary that
        # contains front layers (before permute) and back layers (after permute)
        # for each core network loop

        self.core_layers = nn.ModuleDict()

        # Initialize the first core layer early to insert the first layernorm ahead of the ConvNext Blocks
        self.core_layers["l0_front"] = nn.ModuleList()
        self.core_layers["l0_back"] = nn.ModuleList()

        self.core_layers["l0_front"].append(
            nn.LayerNorm(self.hparams.block_channels[0], eps=1e-6)
        )

        for i in range(len(self.hparams.block_channels)):

            if i != 0:
                self.core_layers[f"l{i}_front"] = nn.ModuleList()
                self.core_layers[f"l{i}_back"] = nn.ModuleList()

            layers_front, layers_back = [], []

            layers_front.extend(
                [
                    ConvNextBlock(
                        self.hparams.block_channels[i],
                        kernel_size=hparams.kernel_size,
                        density=hparams.density,
                    )
                    for block in range(hparams.block_depth[i])
                ]
            )

            if i != len(self.hparams.block_channels) - 1:
                # Append downsampling components

                layers_front.append(
                    nn.LayerNorm(self.hparams.block_channels[i], eps=1e-6)
                )

                layers_back.append(
                    nn.Conv2d(
                        self.hparams.block_channels[i],
                        self.hparams.block_channels[i + 1],
                        kernel_size=2,
                        stride=2,
                    )
                )

            else:
                layers_front.append(
                    nn.LayerNorm(self.hparams.block_channels[i], eps=1e-6)
                )
                layers_back.append(nn.AdaptiveAvgPool2d(output_size=1))

            self.core_layers[f"l{i}_front"].extend(layers_front)
            self.core_layers[f"l{i}_back"].extend(layers_back)

        ### END Core Network Initialization ###

        # Model head
        self.head = nn.Linear(self.hparams.block_channels[-1], classes)

    def forward(self, x, headless=False):
        x = self.initial_layer(x)

        for section_idx in range(len(self.hparams.block_channels)):

            # print(f"section: {section_idx}")

            # Permute before front forward pass
            x = x.permute(0, 2, 3, 1)  # (B,C,Y,X) -> (B,Y,X,C)
            # print(f"permute_f: {x.shape}")

            # front layers forward pass
            for layer in self.core_layers[f"l{section_idx}_front"]:
                x = layer(x)
            # print(f"front: {x.shape}")

            # permute before back layers
            x = x.permute(0, 3, 1, 2)  # (B,Y,X,C) -> (B,C,Y,X)
            # print(f"permute_back: {x.shape}")

            # back layers forward pass
            for layer in self.core_layers[f"l{section_idx}_back"]:
                x = layer(x)
            # print(f"back: {x.shape}")

        x = x.flatten(start_dim=1)

        if headless:
            return x
        return self.head(x)
