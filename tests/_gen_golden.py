"""Generate golden reference outputs from the CURRENT implementations.

Run this once BEFORE refactoring. The test suite then compares the (possibly
refactored) implementations against these snapshots to prove behavior is
unchanged. Re-run only if the functions' behavior is intentionally changed.

Usage:
    python -m tests._gen_golden
"""

from pathlib import Path

import numpy as np
import torch

from losses.triplet_loss import TripletLoss
from tests import inputs
from utils.descriptor_stats import compute_stats
from utils.utils_3D import compute_GT_matches_matrix_3D
from utils.utils_homography import sample_homography

GOLDEN_DIR = Path(__file__).parent / "golden"


def gen_homography() -> None:
    """Snapshot sample_homography for each case."""
    out = {}
    for case in inputs.homography_cases():
        np.random.seed(case["seed"])
        h, h_params = sample_homography(
            case["patch_center"],
            case["patch_shape"],
            case["source_img_shape"],
            case["params"],
        )
        out[case["name"]] = {"H": torch.from_numpy(np.asarray(h, dtype=np.float64))}
    torch.save(out, GOLDEN_DIR / "homography.pt")


def gen_gt_matches() -> None:
    """Snapshot compute_GT_matches_matrix_3D for both flags."""
    data = inputs.gt_matches_inputs()
    out = {}
    for allow in (False, True):
        m, xy0p, xy1p, d0, d1 = compute_GT_matches_matrix_3D(
            **data, return_distances_and_projected=True, allow_multiple_matches=allow
        )
        out[f"allow_{allow}"] = {
            "matrix": m,
            "xy0_proj": xy0p,
            "xy1_proj": xy1p,
            "dist0": d0,
            "dist1": d1,
        }
    torch.save(out, GOLDEN_DIR / "gt_matches.pt")


def gen_triplets() -> None:
    """Snapshot get_hardest_triplets across scale/random regimes."""
    out = {}
    for multiscale in (False, True):
        for ratio in (0.0, 1.0):
            data = inputs.triplet_inputs(multiscale=multiscale)
            loss = TripletLoss(margin=1.0, random_negative_ratio=ratio)
            torch.manual_seed(0)
            triplets = loss.get_hardest_triplets(**data)
            out[f"ms{int(multiscale)}_r{ratio}"] = triplets
    torch.save(out, GOLDEN_DIR / "triplets.pt")


def gen_stats() -> None:
    """Snapshot compute_stats outputs."""
    data = inputs.stats_inputs()
    numeric, tensors = compute_stats(**data)
    torch.save({"numeric": numeric, "tensors": tensors}, GOLDEN_DIR / "stats.pt")


def main() -> None:
    """Generate all golden files."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    gen_homography()
    gen_gt_matches()
    gen_triplets()
    gen_stats()
    print(f"Golden files written to {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
