"""Characterization tests for utils.utils_3D.compute_GT_matches_matrix_3D."""

import pytest
import torch

from tests import inputs
from utils.utils_3D import compute_GT_matches_matrix_3D


@pytest.mark.parametrize("allow", [False, True])
def test_gt_matches_matches_golden(allow: bool, golden: dict) -> None:
    """The GT matrix and intermediates reproduce the pre-refactor outputs."""
    data = inputs.gt_matches_inputs()
    matrix, xy0p, xy1p, d0, d1 = compute_GT_matches_matrix_3D(
        **data, return_distances_and_projected=True, allow_multiple_matches=allow
    )
    ref = golden["gt_matches"][f"allow_{allow}"]
    assert torch.equal(matrix, ref["matrix"])
    torch.testing.assert_close(xy0p, ref["xy0_proj"], equal_nan=True)
    torch.testing.assert_close(xy1p, ref["xy1_proj"], equal_nan=True)
    torch.testing.assert_close(d0, ref["dist0"], equal_nan=True)
    torch.testing.assert_close(d1, ref["dist1"], equal_nan=True)


def test_single_return_equals_matrix_in_tuple(golden: dict) -> None:
    """The bare-matrix return path equals the matrix from the tuple path."""
    data = inputs.gt_matches_inputs()
    matrix = compute_GT_matches_matrix_3D(**data)
    assert torch.equal(matrix, golden["gt_matches"]["allow_False"]["matrix"])


def test_output_shape_and_dtype() -> None:
    """Output has shape B x (n0+1) x (n1+1) and bool dtype."""
    data = inputs.gt_matches_inputs()
    matrix = compute_GT_matches_matrix_3D(**data)
    b, n0 = data["xy0"].shape[:2]
    n1 = data["xy1"].shape[1]
    assert matrix.shape == (b, n0 + 1, n1 + 1)
    assert matrix.dtype == torch.bool
