from __future__ import annotations
import numpy as np
import torch as th
from torch import Tensor


def get_valid_keypoint_mask(
        xy: np.ndarray,
        patch_center: (int, int) | np.ndarray,
        img_size: (int, int) | np.ndarray
) -> np.ndarray:
    """ returns a mask of valid keypoints, i.e. keypoints that are inside the patch
    Args:
        xy: input coordinates (with the convention top-left pixel center at (0.5, 0.5))
        patch_center: center of the patch in the image (x, y)
        img_size: size of the image (H, W)
    """
    mask = \
        (xy[:, 0] > patch_center[0] - img_size[1] // 2) & \
        (xy[:, 0] < patch_center[0] + img_size[1] // 2) & \
        (xy[:, 1] > patch_center[1] - img_size[0] // 2) & \
        (xy[:, 1] < patch_center[1] + img_size[0] // 2)

    return mask


def filter_outside_window(xy: Tensor, top_left_and_bottom_right: Tensor, border: int = 0):
    """ set as nan all the points that are not inside the rectangle defined by top_left_and_bottom_right
    Args:
        xy: keypoints with coordinate (x, y)
            (B),n,2
        top_left_and_bottom_right: top left and bottom right coordinates of the rectangle (x, y)
            (B),2,2
        border: the minimum border to apply when masking
    Returns:
        Tensor: input keypoints with 'nan' where one of the two coordinates was not contained inside shape
        xy_filtered     (B),n,2
    """
    assert xy.shape[-1] == 2, f'the last dimension of xy must be 2, but is {xy.shape[-1]}'
    assert top_left_and_bottom_right.shape[-2:] == (2, 2), \
        f'the last two dims of top_left_and_bottom_right must be 2,2 but are {top_left_and_bottom_right.shape[-2:]}'

    xy = xy.clone()
    tl, br = top_left_and_bottom_right[..., 0, :], top_left_and_bottom_right[..., 1, :]
    outside_mask = (xy[..., 0] < tl[..., 0] + border) + (xy[..., 0] >= br[..., 0] - border) + \
                   (xy[..., 1] < tl[..., 1] + border) + (xy[..., 1] >= br[..., 1] - border)  # (B),n
    xy[outside_mask] = float('nan')
    return xy


def find_mutual_nearest_neighbors_from_keypoints_and_their_projections(
        xy0: Tensor,
        xy1: Tensor,
        xy0_proj: Tensor,
        xy1_proj: Tensor,
        dist0: Tensor | None = None,
        dist1: Tensor | None = None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
    """ find the mutual nearest neighbors between two sets of keypoints and their projections
    Args:
        xy0: first set of keypoints
            n0,2
        xy1: second set of keypoints
            n1,2
        xy0_proj: first set of projected keypoints
            n0_proj,2
        xy1_proj: second set of projected keypoints
            n1_proj,2
        dist0: optional distance matrix between xy0 and xy1_proj, if not provided it will be computed
            n0,n1
        dist1: optional distance matrix between xy0_proj and xy1, if not provided it will be computed
            n0,n1
    Returns:
        mnn_mask: binary mask of mutual nearest neighbors
            n0,n1
        xy0_closest_dist_mnn: for each xy0 that has a mutual nearest neighbor, the distance to the closest xy1_proj in img0
            n0_mnn
        xy1_closest_dist_mnn: for each xy1 that has a mutual nearest neighbor, the distance to the closest xy0_proj in img1
            n1_mnn
        xy0_closest_dist: for each xy0, the distance to the closest xy1_proj in img0
            n0
        xy1_closest_dist: for each xy1, the distance to the closest xy0_proj in img1
            n1
    """
    assert xy0.ndim == 2 and xy1.ndim == 2, f'xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}'
    assert xy0_proj.ndim == 2 and xy1_proj.ndim == 2, f'xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}'
    if dist0 is not None:
        assert dist0.shape == (xy0.shape[0], xy1_proj.shape[0]), \
            f'dist0 must be a matrix of shape ({xy0.shape[0]}, {xy1.shape[0]}), got {dist0.shape}'
    if dist1 is not None:
        assert dist1.shape == (xy0.shape[0], xy1_proj.shape[0]), \
            f'dist1 must be a matrix of shape ({xy0.shape[0]}, {xy1.shape[0]}), got {dist1.shape}'

    device = xy0.device

    n0 = xy0.shape[0]
    n1 = xy1.shape[0]

    if dist0 is None and dist1 is None:
        dist0, dist1 = find_distance_matrices_between_points_and_their_projections(xy0, xy1, xy0_proj, xy1_proj)

    if n1 > 0:
        # ? find the closest point in the image between each xy0 and xy1_proj
        xy0_closest_dist, closest0 = dist0.min(1)
    else:
        xy0_closest_dist, closest0 = th.zeros((0,), device=device), th.zeros((0,), dtype=th.long, device=device)  # n0
    if n0 > 0:
        # ? find the closest point in the image between each xy1 and xy0_proj
        xy1_closest_dist, closest1 = dist1.min(0)
    else:
        xy1_closest_dist, closest1 = th.zeros((0,), device=device), th.zeros((0,), dtype=th.long, device=device)  # n1

    xy0_closest_matrix = th.zeros(dist0.shape, dtype=th.bool, device=device)
    xy1_closest_matrix = th.zeros(dist0.shape, dtype=th.bool, device=device)
    if n1 > 0:
        xy0_closest_matrix[th.arange(len(xy0)), closest0] = True
    if n0 > 0:
        xy1_closest_matrix[closest1, th.arange(len(xy1))] = True
    # ? fink the keypoints that are mutual nearest neighbors (using only x,y coordinates) in both images
    mnn_mask = xy0_closest_matrix & xy1_closest_matrix
    mnn_idx = mnn_mask.nonzero()
    xy0_closest_dist_mnn = th.ones_like(xy0_closest_dist) * float('inf')
    xy1_closest_dist_mnn = th.ones_like(xy1_closest_dist) * float('inf')
    xy0_closest_dist_mnn[mnn_idx[:, 0]] = xy0_closest_dist[mnn_idx[:, 0]]
    xy1_closest_dist_mnn[mnn_idx[:, 1]] = xy1_closest_dist[mnn_idx[:, 1]]
    return mnn_mask, xy0_closest_dist_mnn, xy1_closest_dist_mnn, xy0_closest_dist, xy1_closest_dist


def find_distance_matrices_between_points_and_their_projections(
        xy0: Tensor,
        xy1: Tensor,
        xy0_proj: Tensor,
        xy1_proj: Tensor
) -> (Tensor, Tensor):
    """ find the mutual nearest neighbors between two sets of keypoints and their projections
    Args:
        xy0: first set of keypoints
            n0,2
        xy1: second set of keypoints
            n1,2
        xy0_proj: first set of projected keypoints
            n0_proj,2
        xy1_proj: second set of projected keypoints
            n1_proj,2
    Returns:
        dist0: distance matrix between xy0 and xy1_proj
            n0,n1
        dist1: distance matrix between xy0_proj and xy1
            n0,n1
    """

    assert xy0.ndim == 2 and xy1.ndim == 2, f'xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}'
    assert xy0_proj.ndim == 2 and xy1_proj.ndim == 2, f'xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}'

    # ? compute the distance between all the reprojected points
    # # ? low memory usage, slow but correct
    # dist0 = th.cdist(xy0.to(th.float), xy1_proj,         compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # dist1 = th.cdist(xy0_proj,         xy1.to(th.float), compute_mode='donot_use_mm_for_euclid_dist')  # n0,n1
    # ? high memory usage, fast and correct
    dist0 = (xy0[:, None, :] - xy1_proj[None, :, :]).norm(dim=2)  # n0,n1
    dist1 = (xy0_proj[:, None, :] - xy1[None, :, :]).norm(dim=2)  # n0,n1
    # # ? low memory usage, fast but non-deterministic
    # dist0 = th.cdist(xy0.to(th.float), xy1_proj)  # n0,n1
    # dist1 = th.cdist(xy0_proj,         xy1.to(th.float))  # n0,n1
    dist0[dist0.isnan()] = float('+inf')
    dist1[dist1.isnan()] = float('+inf')
    return dist0, dist1


def nms_keypoints(kpts: Tensor, kpts_score: Tensor, nms_radius: float) -> tuple[Tensor, Tensor]:
    assert kpts.ndim == 2, f'kpts must be a 2D tensor, got {kpts.ndim}'
    assert kpts.shape[-1] == 2, f'kpts must have 2 coordinates, got {kpts.shape[-1]}'
    assert kpts.shape[0] == kpts_score.shape[0], f'kpts and kpts_score must have the same number of keypoints, got {kpts.shape[0]} and {kpts_score.shape[0]}'

    dist = th.cdist(kpts[None], kpts[None])[0]  # n,n
    # dist = th.cdist(kpts[None], kpts[None], compute_mode='donot_use_mm_for_euclid_dist')[0]  # n,n
    dist[th.eye(kpts.shape[0], dtype=th.bool, device=kpts.device)] = float('inf')

    closest_idx = dist.argmin(-1)  # n
    closest_dist = dist[th.arange(kpts.shape[0]), closest_idx]  # n
    closest_score = kpts_score[closest_idx]  # n
    suppression_mask: Tensor = (closest_dist <= nms_radius) * (closest_score > kpts_score)  # n

    kpts_local_maxima = kpts[~suppression_mask]  # n,2

    # print(f'NMS: {kpts.shape[0]} -> {kpts_local_maxima.shape[0]}')
    return kpts_local_maxima, ~suppression_mask

