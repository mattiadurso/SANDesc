"""Characterization tests for TripletLoss.get_hardest_triplets.

Pins the exact triplets tensor for fixed seeds across single/multi-scale GT and
random/hardest negative regimes, guarding the RNG draw order through the refactor.
"""

import pytest
import torch

from losses.triplet_loss import TripletLoss
from tests import inputs

REGIMES = [(False, 0.0), (False, 1.0), (True, 0.0), (True, 1.0)]


@pytest.mark.parametrize("multiscale,ratio", REGIMES)
def test_triplets_match_golden(multiscale: bool, ratio: float, golden: dict) -> None:
    """get_hardest_triplets reproduces the pre-refactor triplets exactly."""
    data = inputs.triplet_inputs(multiscale=multiscale)
    loss = TripletLoss(margin=1.0, random_negative_ratio=ratio)
    torch.manual_seed(0)
    triplets = loss.get_hardest_triplets(**data)
    ref = golden["triplets"][f"ms{int(multiscale)}_r{ratio}"]
    torch.testing.assert_close(triplets, ref, equal_nan=True)


def test_triplet_shape() -> None:
    """Returned triplets have shape (n_triplets, 3, des_dim)."""
    data = inputs.triplet_inputs(multiscale=False)
    loss = TripletLoss(margin=1.0, random_negative_ratio=0.0)
    torch.manual_seed(0)
    triplets = loss.get_hardest_triplets(**data)
    assert triplets.ndim == 3
    assert triplets.shape[1] == 3
    assert triplets.shape[2] == data["des0"].shape[-1]


def test_empty_gt_returns_empty() -> None:
    """No GT matches yields an empty triplets tensor."""
    data = inputs.triplet_inputs(multiscale=False)
    data["matches_matrix_GT_with_bins"] = torch.zeros_like(
        data["matches_matrix_GT_with_bins"]
    )
    loss = TripletLoss(margin=1.0, random_negative_ratio=0.0)
    triplets = loss.get_hardest_triplets(**data)
    assert triplets.numel() == 0
