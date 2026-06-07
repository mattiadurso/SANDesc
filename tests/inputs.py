"""Deterministic input builders shared by golden generation and tests.

These build fixed, seeded inputs for the four functions under refactor so that
their outputs can be characterized (pinned) before and after the refactor.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# sample_homography
# --------------------------------------------------------------------------- #
def homography_cases() -> list[dict]:
    """Return named cases for ``sample_homography`` (params + call kwargs)."""
    base_std = {
        "angle_std": 15.0,
        "angle_p": 1.0,
        "translation_std": 0.1,
        "translation_p": 1.0,
        "shear_std": 0.1,
        "shear_p": 1.0,
        "scale_std": 0.1,
        "scale_p": 1.0,
        "perspective_std": 0.1,
        "perspective_p": 1.0,
    }
    return [
        {
            "name": "std_padding",
            "seed": 0,
            "params": {**base_std, "allow_results_with_padding": True},
            "patch_center": np.array([128, 128]),
            "patch_shape": (64, 64),
            "source_img_shape": (256, 256),
        },
        {
            "name": "std_inside",
            "seed": 42,
            "params": {
                "angle_std": 5.0,
                "angle_p": 1.0,
                "translation_std": 0.02,
                "translation_p": 1.0,
                "shear_std": 0.02,
                "shear_p": 1.0,
                "scale_std": 0.02,
                "scale_p": 1.0,
                "perspective_std": 0.02,
                "perspective_p": 1.0,
                "allow_results_with_padding": False,
            },
            "patch_center": np.array([256, 256]),
            "patch_shape": (32, 32),
            "source_img_shape": (512, 512),
        },
        {
            "name": "no_transforms",
            "seed": 7,
            "params": {
                "angle_std": 15.0,
                "angle_p": 0.0,
                "translation_std": 0.1,
                "translation_p": 0.0,
                "shear_std": 0.1,
                "shear_p": 0.0,
                "scale_std": 0.1,
                "scale_p": 0.0,
                "perspective_std": 0.1,
                "perspective_p": 0.0,
                "allow_results_with_padding": True,
            },
            "patch_center": np.array([128, 128]),
            "patch_shape": (64, 64),
            "source_img_shape": (256, 256),
        },
    ]


# --------------------------------------------------------------------------- #
# compute_GT_matches_matrix_3D
# --------------------------------------------------------------------------- #
def gt_matches_inputs(seed: int = 0) -> dict:
    """Build a small synthetic two-view scene on CPU."""
    g = torch.Generator().manual_seed(seed)
    h = w = 64
    n0 = n1 = 10
    b = 1
    xy0 = torch.rand(b, n0, 2, generator=g) * (w - 1)
    xy1 = torch.rand(b, n1, 2, generator=g) * (w - 1)
    depthmap0 = 2.0 + 3.0 * torch.rand(b, h, w, generator=g)
    depthmap1 = 2.0 + 3.0 * torch.rand(b, h, w, generator=g)
    f, c = 50.0, 32.0
    k = torch.tensor([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])
    k0 = k[None].clone()
    k1 = k[None].clone()
    p0 = torch.eye(4)[None].clone()
    p1 = torch.eye(4)[None].clone()
    p1[0, 0, 3] = 0.5  # small baseline translation in x
    return {
        "xy0": xy0,
        "xy1": xy1,
        "depthmap0": depthmap0,
        "depthmap1": depthmap1,
        "P0": p0,
        "P1": p1,
        "K0": k0,
        "K1": k1,
    }


# --------------------------------------------------------------------------- #
# get_hardest_triplets
# --------------------------------------------------------------------------- #
def _normalized(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def triplet_inputs(multiscale: bool, seed: int = 0) -> dict:
    """Build descriptors and a GT match matrix (single- or multi-scale)."""
    g = torch.Generator().manual_seed(seed)
    b, n0, n1, dim = 1, 6, 6, 8
    des0 = _normalized(torch.randn(b, n0, dim, generator=g))
    des1 = _normalized(torch.randn(b, n1, dim, generator=g))
    matches = torch.zeros(b, n0 + 1, n1 + 1, dtype=torch.bool)
    # diagonal matches
    for i in range(4):
        matches[0, i, i] = True
    if multiscale:
        # extra matches in some rows/columns (more than one per kpt)
        matches[0, 0, 1] = True
        matches[0, 2, 3] = True
    return {"des0": des0, "des1": des1, "matches_matrix_GT_with_bins": matches}


# --------------------------------------------------------------------------- #
# compute_stats
# --------------------------------------------------------------------------- #
def stats_inputs(seed: int = 0) -> dict:
    """Build a descriptor score matrix and a GT match matrix with bins."""
    g = torch.Generator().manual_seed(seed)
    b, n0, n1 = 1, 8, 8
    score_matrix = 2.0 * torch.rand(b, n0, n1, generator=g) - 1.0  # [-1, 1]
    gt = torch.zeros(b, n0 + 1, n1 + 1, dtype=torch.bool)
    for i in range(5):
        gt[0, i, i] = True  # 5 GT matches on the diagonal
    return {
        "score_matrix_des": score_matrix,
        "matching_matrix_GT_with_bins": gt,
    }
