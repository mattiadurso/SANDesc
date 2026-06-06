"""Tests for the SANDesc descriptor network forward pass."""

import pytest
import torch

from model.network_descriptor import SANDesc

# Descriptor dim = last element of the default up_output_channels.
DES_DIM = 128


def test_forward_output_shape() -> None:
    """Output is a dense volume [B, des_dim, H, W] at the input resolution."""
    model = SANDesc().eval()
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, DES_DIM, 32, 32)


def test_forward_is_l2_normalized_by_default() -> None:
    """Descriptors are unit-norm along the channel dim when normalize=True."""
    model = SANDesc().eval()
    out = model(torch.randn(2, 3, 32, 32))
    norms = out.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_normalize_false_skips_normalization() -> None:
    """With normalize=False the output is generally not unit-norm."""
    model = SANDesc().eval()
    out = model(torch.randn(1, 3, 32, 32), normalize=False)
    norms = out.norm(p=2, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("attention", [False, True])
@pytest.mark.parametrize("third", [False, True])
def test_forward_with_config_toggles(skip: bool, attention: bool, third: bool) -> None:
    """All block-config combinations run and keep the output shape finite."""
    model = SANDesc(
        skip_connection=skip, spatial_attention=attention, third_block=third
    ).eval()
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, DES_DIM, 32, 32)
    assert torch.isfinite(out).all()


def test_grayscale_input_channel() -> None:
    """The network honors a custom number of input channels."""
    model = SANDesc(ch_in=1).eval()
    out = model(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, DES_DIM, 32, 32)
