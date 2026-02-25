import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Optional


# --------------------------------------------------------
#                   HELPERS
# --------------------------------------------------------


def get_norm(norm: str, ch_in: int):
    """
    Returns a normalization layer given a string.
    """
    assert norm in [
        "batch",
        "instance",
        "group",
        None,
    ], f"Norm type {norm} not recognized"

    norms = {
        "batch": nn.BatchNorm2d(ch_in),
        "instance": nn.InstanceNorm2d(ch_in, affine=True),
        "group": nn.GroupNorm(num_groups=ch_in // 16, num_channels=ch_in),
        None: nn.Identity(),
    }

    return norms[norm]


def get_activ(activ: str):
    assert activ in ["relu", "gelu", None], f"Activation type {activ} not recognized"
    activations = {
        "relu": nn.ReLU(inplace=False),
        "gelu": nn.GELU(),
        None: nn.Identity(),
    }
    return activations[activ]


# --------------------------------------------------------
#                   UNET MODULES
# --------------------------------------------------------
class Unet_block(nn.Module):
    """
    Pre-activation block for Unet. Similar to DISK.
    Why pre-activ? https://arxiv.org/abs/1603.05027

    """

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        norm: str = "batch",
        activ: str = "relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.norm = get_norm(norm, ch_in)
        self.activ = get_activ(activ)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


class Unet_down_block(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        activ: Optional[str] = "relu",
        norm: str = "batch",
        third_block: bool = False,
        skip_connection: bool = False,
        spatial_attention: bool = False,
    ):
        """
        Args:
            ch_in: int, number of input channels
            ch_out: int, number of output channels
            kernel_size: int, kernel size for the convolutions
            activ: str, activation function
            norm: str, normalization layer
            skip_connection: bool, if True, adds a skip connection. If false same unet as in DISK and S-TREK.
            spatial_attention: bool, if True, adds a spatial attention module. Works only if skip_connection is True.
        """
        super().__init__()
        self.skip_connection = skip_connection

        self.block1 = Unet_block(ch_in, ch_out, kernel_size, norm, activ)
        if skip_connection:
            self.align = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=False)
            self.block2 = Unet_block(ch_out, ch_out, kernel_size, norm, activ)
            self.block3 = (
                Unet_block(ch_out, ch_out, kernel_size, norm, activ)
                if third_block
                else nn.Identity()
            )

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Spatial attention
        self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.pool(x)
        x = self.block1(x_)
        if self.skip_connection:
            x = self.block2(x)  # second block
            x = self.block3(x)  # third block
            x = self.cbam(x)  # spatial attention
            x = self.align(x_) + x  # skip connection
        return x


class Unet_up_block(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 5,
        activ: Optional[str] = None,
        norm: str = "batch",
        third_block: bool = False,
        skip_connection: bool = False,
        spatial_attention: bool = False,
    ):
        super().__init__()
        self.skip_connection = skip_connection
        self.block1 = Unet_block(ch_in, ch_out, kernel_size, norm, activ)
        if skip_connection:
            self.align = nn.Conv2d(
                ch_out, ch_out, kernel_size=1, padding=0, bias=False
            )  # should I add this for the sake of simmetry?
            self.block2 = Unet_block(ch_out, ch_out, kernel_size, norm, activ)
            self.block3 = (
                Unet_block(ch_out, ch_out, kernel_size, norm, activ)
                if third_block
                else nn.Identity()
            )

        self.upsample_2x = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # Spatial attention
        self.cbam = CBAM(gate_channels=ch_out) if spatial_attention else nn.Identity()

    def forward(self, x: Tensor, x_from_past: Tensor = None):
        x_ = self.upsample_2x(x)  # c -> c
        x = torch.cat([x_, x_from_past], dim=1)
        x = self.block1(x)
        if self.skip_connection:
            x = self.block2(x)  # second block
            x = self.block3(x)  # third block
            x = self.cbam(x)  # spatial attention
            x = self.align(x_) + x  # skip connection
        return x


# --------------------------------------------------------
#                   Spatial Attention
# --------------------------------------------------------
# from: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        b, c, h, w = x.size()
        for pool_type in self.pool_types:
            if pool_type == "avg":
                pool = F.adaptive_avg_pool2d(x, 1)
            elif pool_type == "max":
                pool = F.adaptive_max_pool2d(x, 1)
            else:
                continue
            channel_att_raw = self.mlp(pool)
            channel_att_sum = (
                channel_att_raw
                if channel_att_sum is None
                else channel_att_sum + channel_att_raw
            )

        scale = torch.sigmoid(channel_att_sum).view(b, c, 1, 1)
        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)),
            dim=1,
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2,
            1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False,
            bn=True,
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        use_spatial=True,
    ):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate() if use_spatial else nn.Identity()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
