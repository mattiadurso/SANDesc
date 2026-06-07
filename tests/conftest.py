"""Shared test fixtures: load golden reference snapshots."""

from pathlib import Path

import pytest
import torch

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture(scope="session")
def golden() -> dict:
    """Load all golden snapshots produced by tests/_gen_golden.py."""
    return {
        name: torch.load(GOLDEN_DIR / f"{name}.pt", weights_only=False)
        for name in ("homography", "gt_matches", "triplets", "stats")
    }
