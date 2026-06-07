"""Characterization tests for utils.descriptor_stats.compute_stats."""

import math

import torch

from tests import inputs
from utils.descriptor_stats import compute_stats


def test_numeric_stats_match_golden(golden: dict) -> None:
    """Every numeric stat reproduces the pre-refactor value."""
    data = inputs.stats_inputs()
    numeric, _ = compute_stats(**data)
    ref = golden["stats"]["numeric"]
    assert numeric.keys() == ref.keys()
    for key, value in ref.items():
        got = numeric[key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(got), f"{key}: expected nan, got {got}"
        else:
            assert got == value or math.isclose(
                got, value, rel_tol=1e-6, abs_tol=1e-6
            ), f"{key}: {got} != {value}"


def test_tensor_stats_match_golden(golden: dict) -> None:
    """Every tensor stat reproduces the pre-refactor tensor."""
    data = inputs.stats_inputs()
    _, tensors = compute_stats(**data)
    ref = golden["stats"]["tensors"]
    assert tensors.keys() == ref.keys()
    for key, value in ref.items():
        torch.testing.assert_close(tensors[key], value, equal_nan=True)


def test_precision_recall_in_range() -> None:
    """Precision and recall are valid probabilities for this input."""
    data = inputs.stats_inputs()
    numeric, _ = compute_stats(**data)
    assert 0.0 <= numeric["matches_precision"] <= 1.0
    assert 0.0 <= numeric["matches_recall"] <= 1.0
