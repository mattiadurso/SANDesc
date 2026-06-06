"""SANDesc UNet-style descriptor network."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add project root and external repos to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from model.modules import UNetDownBlock, UNetUpBlock  # noqa: E402


class SANDesc(nn.Module):
    """UNet-style encoder-decoder producing a dense descriptor volume."""

    def __init__(
        self,
        ch_in: int = 3,
        kernel_size: int = 5,
        activ: str = "gelu",
        norm: str = "batch",
        skip_connection: bool = False,
        spatial_attention: bool = False,
        third_block: bool = False,
        down_output_channels: list[int] | None = None,
        up_output_channels: list[int] | None = None,
        **kwargs: object,
    ) -> None:
        """Build the descriptor network.

        The last element of ``up_output_channels`` is the descriptor dimension.

        Args:
            ch_in: Number of input channels.
            kernel_size: Kernel size of the convolutional layers.
            activ: Activation function: 'relu', 'prelu' or 'gelu'.
            norm: Normalization layer type.
            skip_connection: If True, add skip connections and a second unet
                block to the network.
            spatial_attention: If True, add spatial attention to the network.
            third_block: If True, add a third unet block to the network.
            down_output_channels: Output channels of each down block.
            up_output_channels: Output channels of each up block. Add +1 to the
                last element to match the DISK unet, e.g. [64, 64, 64, 128 + 1].
            **kwargs: Ignored extra keyword arguments.
        """
        super().__init__()
        if down_output_channels is None:
            down_output_channels = [16, 32, 64, 64, 64]
        if up_output_channels is None:
            up_output_channels = [64, 64, 64, 128]
        self.conv_highest = nn.Conv2d(
            ch_in,
            down_output_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

        common = {
            "kernel_size": kernel_size,
            "activ": activ,
            "norm": norm,
            "skip_connection": skip_connection,  # and second block
            "spatial_attention": spatial_attention,
            "third_block": third_block,
        }

        self.down0 = UNetDownBlock(
            down_output_channels[0], down_output_channels[1], **common
        )
        self.down1 = UNetDownBlock(
            down_output_channels[1], down_output_channels[2], **common
        )
        self.down2 = UNetDownBlock(
            down_output_channels[2], down_output_channels[3], **common
        )
        self.down3 = UNetDownBlock(
            down_output_channels[3], down_output_channels[4], **common
        )

        self.up0 = UNetUpBlock(
            down_output_channels[-1] + down_output_channels[-2],
            up_output_channels[0],
            **common,
        )
        self.up1 = UNetUpBlock(
            down_output_channels[-3] + up_output_channels[0],
            up_output_channels[1],
            **common,
        )
        self.up2 = UNetUpBlock(
            down_output_channels[-4] + up_output_channels[1],
            up_output_channels[2],
            **common,
        )
        self.up3 = UNetUpBlock(
            down_output_channels[-5] + up_output_channels[2],
            up_output_channels[3],
            kernel_size=kernel_size,
            activ=None,
            norm=None,
        )

    def load_weights(self, weights: str) -> None:
        """Load weights into the model.

        Args:
            weights (str): Path to the weights file.

        """
        weights = torch.load(weights, weights_only=False)
        self.load_state_dict(weights["state_dict"])

    def forward(
        self, x: Tensor, _: Tensor | None = None, normalize: bool = True
    ) -> Tensor:
        """Compute the dense descriptor volume [B, des_dim, H, W] for input x."""
        x0 = self.conv_highest(x)  # B,c_in,H,W

        x1 = self.down0(x0)  # B,C1,H/2,W/2
        x2 = self.down1(x1)  # B,C2,H/4,W/4
        x3 = self.down2(x2)  # B,C3,H/8,W/8
        x4 = self.down3(x3)  # B,C4,H/16,W/16

        x5 = self.up0(x4, x3)  # B,C5,H/8,W/8
        x6 = self.up1(x5, x2)  # B,C6,H/4,W/4
        x7 = self.up2(x6, x1)  # B,C7,H/2,W/2
        x8 = self.up3(x7, x0)  # B,des_dim,H,W

        if normalize:
            x8 = F.normalize(x8, p=2, dim=1)  # B,des_dim,H,W

        return x8


# main
if __name__ == "__main__":
    from torchinfo import summary

    device = "cuda"

    model = (
        SANDesc(skip_connection=True, spatial_attention=True, third_block=True)
        .to(device)
        .eval()
    )
    summary(model)

    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)

    print(out.shape)
