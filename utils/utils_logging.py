from typing import Optional

import torch as th
from matplotlib import pyplot as plt
from torch import Tensor

from utils.utils_2D import (
    compute_correct_wrong_mismatched_inexistent_unsure_matches,
    mutual_nearest_neighbors_from_score_matrix,
)
from utils.utils_visualization import plot_image_pair_with_keypoints
import wandb


@th.no_grad()
def log_match_plot(
    img0: Tensor,
    img1: Tensor,
    kpts0: Tensor,
    kpts1: Tensor,
    score_matrix: Optional[Tensor],
    matching_matrix_GT_with_bins: Optional[Tensor],
    batch_idx: int,
    iteration: int,
    tag: str = "",
    caption: str = "img",
):
    assert img0.shape[0] == 1, f"img0.shape[0] == {img0.shape[0]} != 1"

    if score_matrix is not None:
        matching_matrix_des = mutual_nearest_neighbors_from_score_matrix(
            score_matrix
        )  # 1,n_kpts0,n_kpts1
        matching_matrix_agg = (
            compute_correct_wrong_mismatched_inexistent_unsure_matches(
                matching_matrix_des, matching_matrix_GT_with_bins
            )
        )
        matching_matrix_agg = matching_matrix_agg[0]

        # ? find the matches color
        matches_correct_idxs = matching_matrix_agg.correct.nonzero()
        matches_mismatch_idxs = matching_matrix_agg.mismatched.nonzero()
        matches_inexistent_idxs = matching_matrix_agg.inexistent.nonzero()
        matches_unsure_idxs = matching_matrix_agg.unsure.nonzero()

        matches_correct_color = th.tensor([0.2, 0.8, 0.2])  # lime
        matches_mismatch_color = th.tensor([1.0, 0.8, 0.0])  # orange
        matches_inexistent_color = th.tensor([1.0, 0.0, 1.0])  # magenta
        matches_unsure_color = th.tensor([0.0, 0.0, 1.0])  # blue

        matches = th.cat(
            (
                matches_correct_idxs,
                matches_mismatch_idxs,
                matches_inexistent_idxs,
                matches_unsure_idxs,
            ),
            dim=0,
        )
        matches_color = th.cat(
            (
                matches_correct_color.repeat(matches_correct_idxs.shape[0], 1),
                matches_mismatch_color.repeat(matches_mismatch_idxs.shape[0], 1),
                matches_inexistent_color.repeat(matches_inexistent_idxs.shape[0], 1),
                matches_unsure_color.repeat(matches_unsure_idxs.shape[0], 1),
            ),
            dim=0,
        )
    else:
        matches = th.zeros((0, 2), dtype=th.long, device=img0.device)
        matches_color = th.zeros((0, 3), dtype=th.float, device=img0.device)

    fig, axes = plot_image_pair_with_keypoints(
        img0[0], img1[0], kpts0[0], kpts1[0], matches, matches_color
    )
    if wandb.run is not None:
        wandb.log(
            {f"{tag}imgs_{batch_idx}": [wandb.Image(fig, caption=caption)]},
            step=iteration,
        )

    plt.close(fig)
