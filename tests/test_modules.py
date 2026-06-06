"""Tests for the reusable network building blocks in model/modules.py."""

import pytest
import torch
from torch import nn

from model.modules import (
    CBAM,
    UNetBlock,
    UNetDownBlock,
    UNetUpBlock,
    get_activ,
    get_norm,
)


def test_get_norm_types() -> None:
    """get_norm maps strings to the expected layers and None to Identity."""
    assert isinstance(get_norm("batch", 16), nn.BatchNorm2d)
    assert isinstance(get_norm("instance", 16), nn.InstanceNorm2d)
    assert isinstance(get_norm("group", 32), nn.GroupNorm)
    assert isinstance(get_norm(None, 16), nn.Identity)


def test_get_activ_types() -> None:
    """get_activ maps strings to the expected activations and None to Identity."""
    assert isinstance(get_activ("relu"), nn.ReLU)
    assert isinstance(get_activ("gelu"), nn.GELU)
    assert isinstance(get_activ(None), nn.Identity)


def test_get_norm_unknown_raises() -> None:
    """An unknown norm string raises ValueError."""
    with pytest.raises(ValueError, match="not recognized"):
        get_norm("unknown", 16)


def test_unet_block_preserves_spatial_changes_channels() -> None:
    """UNetBlock keeps H,W (padded conv) and maps ch_in -> ch_out."""
    block = UNetBlock(ch_in=8, ch_out=16, kernel_size=5)
    out = block(torch.randn(2, 8, 16, 16))
    assert out.shape == (2, 16, 16, 16)


@pytest.mark.parametrize("skip", [False, True])
def test_unet_down_block_halves_spatial(skip: bool) -> None:
    """UNetDownBlock downsamples by 2 and outputs ch_out channels."""
    block = UNetDownBlock(ch_in=8, ch_out=16, skip_connection=skip)
    out = block(torch.randn(2, 8, 16, 16))
    assert out.shape == (2, 16, 8, 8)


def test_unet_up_block_doubles_spatial_with_skip_concat() -> None:
    """UNetUpBlock upsamples x by 2, concatenates the skip, outputs ch_out."""
    # ch_in = upsampled-x channels (16) + skip channels (8) = 24
    block = UNetUpBlock(ch_in=24, ch_out=12, skip_connection=False)
    x = torch.randn(2, 16, 8, 8)
    skip = torch.randn(2, 8, 16, 16)
    out = block(x, skip)
    assert out.shape == (2, 12, 16, 16)


def test_cbam_preserves_shape() -> None:
    """CBAM applies attention without changing the tensor shape."""
    cbam = CBAM(gate_channels=16)
    x = torch.randn(2, 16, 8, 8)
    assert cbam(x).shape == x.shape
