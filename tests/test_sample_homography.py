"""Characterization tests for utils.utils_homography.sample_homography.

Pins the exact homography produced for fixed seeds/params so that the refactor
(splitting the function into helpers) provably preserves behavior, including the
order and number of RNG draws.
"""

import numpy as np
import pytest

from tests import inputs
from utils.utils_homography import sample_homography

CASES = inputs.homography_cases()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_sample_homography_matches_golden(case: dict, golden: dict) -> None:
    """sample_homography reproduces the pre-refactor homography exactly."""
    np.random.seed(case["seed"])
    h, _ = sample_homography(
        case["patch_center"],
        case["patch_shape"],
        case["source_img_shape"],
        case["params"],
    )
    expected = golden["homography"][case["name"]]["H"].numpy()
    got = np.asarray(h, dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_returns_3x3_and_invertible() -> None:
    """A sampled homography is a 3x3 invertible matrix."""
    case = CASES[0]
    np.random.seed(case["seed"])
    h, params = sample_homography(
        case["patch_center"],
        case["patch_shape"],
        case["source_img_shape"],
        case["params"],
    )
    assert h.shape == (3, 3)
    assert abs(np.linalg.det(h)) > 1e-9
    assert isinstance(params, dict)
