"""Functions that works with geometry on 2D images"""

from dataclasses import dataclass

import math

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional


def rotatedRectWithMaxArea(w, h, angle):
    """
    from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr, (w - wr) / 2, (h - hr) / 2


def compute_quadrilateral_area_from_corners(corners):
    area = 0.5 * (
        (
            corners[0, 0] * corners[1, 1]
            + corners[1, 0] * corners[2, 1]
            + corners[2, 0] * corners[3, 1]
            + corners[3, 0] * corners[0, 1]
        )
        - (
            corners[1, 0] * corners[0, 1]
            + corners[2, 0] * corners[1, 1]
            + corners[3, 0] * corners[2, 1]
            + corners[0, 0] * corners[3, 1]
        )
    )
    area = abs(area)
    return area


def grid_sample_nan(xy: Tensor, img: Tensor, mode="nearest") -> Tuple[Tensor, Tensor]:
    """pytorch grid_sample with embedded coordinate normalization and grid nan handling (if a nan is present in xy,
    the output will be nan). Works both with input with shape Bxnx2 and B x n0 x n1 x 2
    xy point that fall outside the image are treated as nan (those which are really close are interpolated using
    border padding mode)
    Args:
        xy: input coordinates (with the convention top-left pixel center at (0.5, 0.5))
            B,n,2 or B,n0,n1,2
        img: the image where the sampling is done
            BxCxHxW or BxHxW
        mode: the interpolation mode
    Returns:
        sampled: the sampled values
            BxCxN or BxCxN0xN1 (if no C dimension in input BxN or BxN0xN1)
        mask_img_nan: mask of the points that had a nan in the img. The points xy that were nan appear as false in the
            mask in the same way as point that had a valid img value. This is done to discriminate between invalid
            sampling position and valid sampling position with a nan value in the image
            BxN or BxN0xN1
    """
    assert img.dim() in {3, 4}
    if img.dim() == 3:
        # remove the channel dimension from the result at the end of the function
        squeeze_result = True
        img.unsqueeze_(1)
    else:
        squeeze_result = False

    assert xy.shape[-1] == 2
    assert (
        xy.dim() == 3 or xy.dim() == 4
    ), f"xy must have 3 or 4 dimensions, got {xy.dim()}"
    B, C, H, W = img.shape

    xy_norm = normalize_pixel_coordinates(xy, img.shape[-2:])  # BxNx2 or BxN0xN1x2
    # set to nan the point that fall out of the second image
    xy_norm[(xy_norm < -1) + (xy_norm > 1)] = float("nan")
    if xy.ndim == 3:
        sampled = F.grid_sample(
            img,
            xy_norm[:, :, None, ...],
            align_corners=False,
            mode=mode,
            padding_mode="border",
        ).view(
            B, C, xy.shape[1]
        )  # BxCxN
    else:
        sampled = F.grid_sample(
            img, xy_norm, align_corners=False, mode=mode, padding_mode="border"
        )  # BxCxN0xN1
    # points xy that are not nan and have nan img. The sum is just to squash the channel dimension
    mask_img_nan = torch.isnan(sampled.sum(1))  # BxN or BxN0xN1
    # set to nan the sampled values for points xy that were nan (grid_sample consider those as (-1, -1))
    xy_invalid = xy_norm.isnan().any(-1)  # BxN or BxN0xN1
    if xy.ndim == 3:
        sampled[xy_invalid[:, None, :].repeat(1, C, 1)] = float("nan")
    else:
        sampled[xy_invalid[:, None, :, :].repeat(1, C, 1, 1)] = float("nan")

    if squeeze_result:
        img.squeeze_(1)
        sampled.squeeze_(1)

    return sampled, mask_img_nan


def normalize_pixel_coordinates(xy: Tensor, shape: Tuple[int, int]) -> Tensor:
    """normalize pixel coordinates from -1 to +1. Being (-1,-1) the exact top left corner of the image
    the coordinates must be given in a way that the center of pixel is at half coordinates (0.5,0.5)
    xy ordered as (x, y) and shape ordered as (H, W)
    Args:
        xy: input coordinates in order (x,y) with the convention top-left pixel center is at coordinates (0.5, 0.5)
            ...x2
        shape: shape of the image in the order (H, W)
    Returns:
        xy_norm: normalized coordinates between [-1, 1]
    """
    xy_norm = xy.clone()
    # ! the shape index are flipped because the coordinates are given as x,y but shape is H,W
    xy_norm[..., 0] = 2 * xy_norm[..., 0] / shape[1]
    xy_norm[..., 1] = 2 * xy_norm[..., 1] / shape[0]
    xy_norm -= 1
    return xy_norm


def filter_outside(xy: Tensor, shape: Tuple[int, int], border: int = 0):
    """set as nan all the points that are not inside rectangle defined with shape HxW
    Args:
        xy: keypoints with coordinate (x, y)
            (B)xnx2
        shape: shape where the keypoints should be contained (H, W)
            2
        border: the minimum border to apply when masking
    Returns:
        Tensor: input keypoints with 'nan' where one of the two coordinates was not contained inside shape
        xy_filtered     (B)xnx2
    """
    assert xy.shape[-1] == 2
    assert border < max(shape)

    xy[(xy[..., 0] < border) + (xy[..., 0] >= shape[1] - border)] = float("nan")
    xy[(xy[..., 1] < border) + (xy[..., 1] >= shape[0] - border)] = float("nan")
    return xy


def find_mutual_nearest_neighbors_from_keypoints_and_their_projections(
    xy0: Tensor, xy1: Tensor, xy0_proj: Tensor, xy1_proj: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}"

    device = xy0.device

    n0 = xy0.shape[0]
    n1 = xy1.shape[0]
    dist0, dist1 = find_distance_matrices_between_points_and_their_projections(
        xy0, xy1, xy0_proj, xy1_proj
    )

    if n1 > 0:
        xy0_closest_dist, closest0 = dist1.min(1)
    else:
        xy0_closest_dist, closest0 = torch.zeros((0,), device=device), torch.zeros(
            (0,), dtype=torch.long, device=device
        )  # n0
    if n0 > 0:
        xy1_closest_dist, closest1 = dist0.min(0)
    else:
        xy1_closest_dist, closest1 = torch.zeros((0,), device=device), torch.zeros(
            (0,), dtype=torch.long, device=device
        )  # n1

    xy0_closest_matrix = torch.zeros(dist0.shape, dtype=torch.bool, device=device)
    xy1_closest_matrix = torch.zeros(dist0.shape, dtype=torch.bool, device=device)
    if n1 > 0:
        xy0_closest_matrix[torch.arange(len(xy0)), dist1.argmin(1)] = True
    if n0 > 0:
        xy1_closest_matrix[dist0.argmin(0), torch.arange(len(xy1))] = True
    # fink the keypoints that are mutual nearest neighbors (using only x,y coordinates) in both images
    mnn_mask = xy0_closest_matrix & xy1_closest_matrix
    mnn_idx = mnn_mask.nonzero()
    xy0_closest_dist_mnn = torch.ones_like(xy0_closest_dist) * float("inf")
    xy1_closest_dist_mnn = torch.ones_like(xy1_closest_dist) * float("inf")
    xy0_closest_dist_mnn[mnn_idx[:, 0]] = xy0_closest_dist[mnn_idx[:, 0]]
    xy1_closest_dist_mnn[mnn_idx[:, 1]] = xy1_closest_dist[mnn_idx[:, 1]]
    return (
        mnn_mask,
        xy0_closest_dist_mnn,
        xy1_closest_dist_mnn,
        xy0_closest_dist,
        xy1_closest_dist,
    )


def find_distance_matrices_between_points_and_their_projections(
    xy0: Tensor, xy1: Tensor, xy0_proj: Tensor, xy1_proj: Tensor
) -> Tuple[Tensor, Tensor]:
    assert (
        xy0.ndim == 2 and xy1.ndim == 2
    ), f"xy0 and xy1 must be 2D tensors, got {xy0.ndim} and {xy1.ndim}"
    assert (
        xy0_proj.ndim == 2 and xy1_proj.ndim == 2
    ), f"xy0_proj and xy1_proj must be 2D tensors, got {xy0_proj.ndim} and {xy1_proj.ndim}"

    # distance between all the reprojected points
    dist0 = torch.cdist(
        xy0.to(torch.float), xy1_proj, compute_mode="donot_use_mm_for_euclid_dist"
    )  # n0 x n1
    dist1 = torch.cdist(
        xy0_proj, xy1.to(torch.float), compute_mode="donot_use_mm_for_euclid_dist"
    )  # n0 x n1
    dist0[dist0.isnan()] = float("+inf")
    dist1[dist1.isnan()] = float("+inf")
    return dist0, dist1


def mutual_nearest_neighbors_from_score_matrix(
    score_mat: Tensor, min_score: float = -1.0, ratio_test: float = 1.0
) -> Tensor:
    """return a boolean matrix with a True where the position was a maximum in both row and columns (and grater than the min score)
    Args:
        score_mat: score_matrix matrix
            Bxn0xn1
        min_score: minimum score to consider a match
        ratio_test: ratio test to apply to the score matrix
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 torch.bool
    """
    assert score_mat.ndim == 3
    B, n0, n1 = score_mat.shape
    if n0 == 0 or n1 == 0:
        return score_mat.new_zeros((B, n0, n1), dtype=torch.bool)

    device = score_mat.device
    score_mat = score_mat.clone()
    score_mat[score_mat.isnan()] = float("-inf")

    # get the closest ones for each row and column
    # each row is the score between the a descriptor from img0 and all the others in img1
    # each column is the score between the a descriptor from img1 and all the others in img0
    nn0_value, nn0_idx = score_mat.max(2)  # (B,n0) (B,n0) with values [0, n1[
    nn1_value, nn1_idx = score_mat.max(1)  # (B,n1) (B,n1) with values [0, n0[

    nn0_idx[nn0_value == float("-inf")] = n1  # (B,n0) with values [0, n1]
    nn1_idx[nn1_value == float("-inf")] = n0  # (B,n1) with values [0, n0]

    nn0_matrix = torch.zeros(
        (B, n0 + 1, n1 + 1), dtype=torch.bool, device=device
    )  # Bxn0xn1
    nn0_matrix[:, :-1, :].scatter_(2, nn0_idx[:, :, None], True)

    nn1_matrix = torch.zeros(
        (B, n0 + 1, n1 + 1), dtype=torch.bool, device=device
    )  # Bxn0xn1
    nn1_matrix[:, :, :-1].scatter_(1, nn1_idx[:, None, :], True)

    # compose the two matrices
    mnn_matrix = nn0_matrix * nn1_matrix

    # drop the bins
    mnn_matrix = mnn_matrix[:, :-1, :-1]

    # remove the ones with score less than the min score
    mnn_matrix = mnn_matrix * (score_mat > min_score)

    if ratio_test < 1.0:
        best_scores0, idxs0 = score_mat.topk(
            2, dim=-1, largest=True, sorted=True
        )  # (B,n0,2) (B,n0,2)
        best_scores1, idxs1 = score_mat.topk(
            2, dim=-2, largest=True, sorted=True
        )  # (B,2,n1) (B,2,n1)
        valid_mask0 = best_scores0[:, :, 0] * ratio_test > best_scores0[:, :, 1]  # B,n0
        valid_mask1 = best_scores1[:, 0, :] * ratio_test > best_scores1[:, 1, :]  # n1
        ratio_test_mat = valid_mask0[:, :, None] * valid_mask1[:, None, :]
        mnn_matrix *= ratio_test_mat

    return mnn_matrix


def mutual_nearest_neighbors_from_dist_matrices(dist0: Tensor, dist1: Tensor) -> Tensor:
    """return a boolean matrix with a True where the position was a minimum in both columns of dist0 and rows of dist1
    Args:
        dist0: distance matrix computed in img0 between rows xy0 and columns xy1_proj
            Bxn0xn1
        dist1: distance matrix computed in img1 between rows xy0_proj and columns xy1
            Bxn0xn1
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 torch.bool
    Raises:
        None
    """
    assert dist0.shape == dist1.shape, "The two matrices must have the same dimensions"
    device = dist0.device

    B, n0, n1 = dist0.shape
    if n0 == 0 or n1 == 0:
        return dist0.new_zeros((B, n0, n1), dtype=torch.bool)

    # get the closest ones for each row and column
    dst0, idx0 = dist0.min(dim=2, keepdim=True)  # (B,n0,1), (B,n0,1)
    dst1, idx1 = dist1.min(dim=1, keepdim=True)  # (B,1,n1), (B,1,n1)
    # set the indexes for infinite distance such that the scatter puts a one in the bin
    idx0[dst0 == float("+inf")] = n1
    idx1[dst1 == float("+inf")] = n0

    closest0_matrix = torch.zeros(
        (B, n0 + 1, n1 + 1), dtype=torch.bool, device=device
    )  # B,n0+1,n1+1
    closest1_matrix = torch.zeros(
        (B, n0 + 1, n1 + 1), dtype=torch.bool, device=device
    )  # B,n0+1,n1+1

    # build the closest one matrix for each kpts0 (every row is dist from a kpts0_i and all the others kpts1)
    # build the closest one matrix for each kpts1 (every row is dist from a kpts1_i and all the others kpts0)
    closest0_matrix.scatter_(2, idx0, True)
    closest1_matrix.scatter_(1, idx1, True)

    # by multiplying the two matrices only the mutual-nearest-neighbours are selected
    mnn_matrix = closest0_matrix * closest1_matrix

    mnn_matrix = mnn_matrix[:, :-1, :-1]

    return mnn_matrix


@dataclass
class MatchingMatrix:
    proposed: torch.Tensor
    correct: torch.Tensor
    wrong: torch.Tensor
    mismatched: torch.Tensor
    inexistent: torch.Tensor
    unsure: torch.Tensor
    score: Optional[torch.Tensor] = None

    def __getitem__(self, b: int = 0):
        return MatchingMatrix(
            proposed=self.proposed[b],
            correct=self.correct[b],
            wrong=self.wrong[b],
            mismatched=self.mismatched[b],
            inexistent=self.inexistent[b],
            unsure=self.unsure[b],
            score=self.score[b] if self.score is not None else None,
        )


def compute_correct_wrong_mismatched_inexistent_unsure_matches(
    matching_matrix: Tensor, GT_matching_matrix_with_bins: Tensor
) -> MatchingMatrix:
    """
    Args:
        matching_matrix: the matching matrix obtained from descriptors
            B,n0,n1
        GT_matching_matrix_with_bins: the GT matching matrix with one additional bin row and column with the unmatched keypoints
            B,n0+1,n1+1

    Returns:
        matching_matrix_correct: the matching matrix with the correct matches
        matching_matrix_wrong: the matching matrix with the wrong matches
        matching_matrix_mismatched: the matching matrix with the mismatched matches (there exist a correct matches for that
            point but its wrongly matched)
        matching_matrix_inexistent: a match is found between two points that have no existing match in the GT_matching_matrix
    """
    assert (
        matching_matrix.shape[0] == GT_matching_matrix_with_bins.shape[0]
    ), f"{matching_matrix.shape[0]} != {GT_matching_matrix_with_bins.shape[0]}"
    assert (
        matching_matrix.shape[1] == GT_matching_matrix_with_bins.shape[1] - 1
    ), f"{matching_matrix.shape[1]} != {GT_matching_matrix_with_bins.shape[1] - 1}"
    assert (
        matching_matrix.shape[2] == GT_matching_matrix_with_bins.shape[2] - 1
    ), f"{matching_matrix.shape[2]} != {GT_matching_matrix_with_bins.shape[2] - 1}"
    assert matching_matrix.ndim == 3, f"{matching_matrix.ndim} != 3"
    assert (
        matching_matrix.dtype == torch.bool
        and GT_matching_matrix_with_bins.dtype == torch.bool
    ), f"{matching_matrix.dtype} != {torch.bool} or {GT_matching_matrix_with_bins.dtype} != {torch.bool}"

    GT_matching_matrix = GT_matching_matrix_with_bins[:, :-1, :-1]
    B, H, W = GT_matching_matrix.shape

    matching_matrix_correct = matching_matrix * GT_matching_matrix

    # known_mask is true for each row and column where there is a one, either as match or in the bin
    known_mask_with_bins = GT_matching_matrix_with_bins.any(1, keepdim=True).repeat(
        1, H + 1, 1
    ) + GT_matching_matrix_with_bins.any(2, keepdim=True).repeat(1, 1, W + 1)
    known_mask = known_mask_with_bins[:, :-1, :-1]

    # match_mask is true for each row and column where there is a GT match
    any_match_mask = GT_matching_matrix.any(1, keepdim=True).repeat(
        1, H, 1
    ) + GT_matching_matrix.any(2, keepdim=True).repeat(1, 1, W)

    # matching_matrix_unsure is true when one of the proposed matches does not correspond either to a match or to an unmatch
    matching_matrix_unsure = matching_matrix * ~known_mask

    # matching_matrix_wrong is true when a proposed match is wrong (either a mismatch or inexistent)
    matching_matrix_wrong = (
        (matching_matrix ^ GT_matching_matrix) * matching_matrix
    ) * known_mask

    # mismatch_mask is true when a point that actually had a possible correct match is mismatched
    matching_matrix_mismatched = matching_matrix_wrong * any_match_mask

    # inexistent_mask is true when two keypoints that had not GT match are matched
    matching_matrix_inexistent = matching_matrix_wrong * ~any_match_mask

    output = MatchingMatrix(
        matching_matrix,
        matching_matrix_correct,
        matching_matrix_wrong,
        matching_matrix_mismatched,
        matching_matrix_inexistent,
        matching_matrix_unsure,
    )
    return output


def add_bins_to_matching_matrix(matching_matrix: Tensor) -> Tensor:
    """
    Args:
        matching_matrix:
            B,n0,n1
    Returns:
        matching_matrix_with_bins:
            B,n0+1,n1+1
    """
    B, H, W = matching_matrix.shape
    matching_matrix_with_bins = torch.zeros(
        (B, H + 1, W + 1), device=matching_matrix.device, dtype=torch.bool
    )
    matching_matrix_with_bins[:, :-1, :-1] = matching_matrix.clone()
    matching_matrix_with_bins[:, :-1, -1] = ~matching_matrix.any(-1)
    matching_matrix_with_bins[:, -1, :-1] = ~matching_matrix.any(1)
    return matching_matrix_with_bins


def compute_corner_mask(x: Tensor, dilation: int = 1, edge_thr: int = 10) -> Tensor:
    print("this function is unchecked COMPUTE_CORNER_MASK")
    # x.shape = B, H, W
    dii_kernel = x.new_tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]])[None, None]
    dij_kernel = x.new_tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])[None, None]
    djj_kernel = x.new_tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]])[None, None]

    dii = F.conv2d(x[:, None, :, :], dii_kernel, padding=1, dilation=dilation)[
        :, 0, :, :
    ]
    dij = F.conv2d(x[:, None, :, :], dij_kernel, padding=1, dilation=dilation)[
        :, 0, :, :
    ]
    djj = F.conv2d(x[:, None, :, :], djj_kernel, padding=1, dilation=dilation)[
        :, 0, :, :
    ]

    det = dii * djj - dij**2
    tr = dii + djj
    thr = (edge_thr + 1) ** 2 / edge_thr
    corner_mask = (tr**2 / det < thr) * (det > 0)

    # # non-edge
    # dii_filter = tf.reshape(tf.constant([[0, 1., 0], [0, -2., 0], [0, 1., 0]]), (3, 3, 1, 1))
    # dij_filter = tf.reshape(0.25 * tf.constant([[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]), (3, 3, 1, 1))
    # djj_filter = tf.reshape(tf.constant([[0, 0, 0], [1., -2., 1.], [0, 0, 0]]), (3, 3, 1, 1))

    # dii_filter = tf.tile(dii_filter, (1, 1, n_channel, 1))

    # pad_inputs = tf.pad(inputs, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]], constant_values=0)

    # dii = tf.nn.depthwise_conv2d(pad_inputs, filter=dii_filter, strides=[1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

    # dij_filter = tf.tile(dij_filter, (1, 1, n_channel, 1))
    # dij = tf.nn.depthwise_conv2d(pad_inputs, filter=dij_filter, strides=[1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

    # djj_filter = tf.tile(djj_filter, (1, 1, n_channel, 1))
    # djj = tf.nn.depthwise_conv2d(pad_inputs, filter=djj_filter, strides=[1, 1, 1, 1], padding='VALID', dilations=[dilation] * 2)

    # det = dii * djj - dij * dij
    # tr = dii + djj
    # thld = (edge_thr + 1)**2 / edge_thr
    # is_not_edge = tf.logical_and(tr * tr / det <= thld, det > 0)
    return corner_mask


def extract_maxima_from_map(
    score_map: Tensor,
    thr: float = 0.1,
    nms_radius: int = 3,
    border: int = 0,
    max_kpts: int = float("inf"),
    edge_thr: int = 0,
) -> Tuple[List[Tensor], List[Tensor]]:
    """extract the keypoints from the provided map (using the convention top-left pixel center (0.5, 0.5) using the
        provided parameters
    Args:
        score_map: the input map where to look for maxima
            BxHxW or Bx1xHxW
        thr: the maxima threshold, everything below this values is not extracted
        nms_radius: the Non-Maxima-Suppression radius
        border: border width to remove from each side
        max_kpts: the max number of kpts to extract (the ones with top scores are selected)
        edge_thr: avoid to extract points from the edges
    Returns:
        kpts: List[Tensor]: the extracted keypoints for every image in the batch ordered by score
            len(kpts) = B   kpts.shape = Nb x 2 (x, y)  with convention top-left pixel center (0.5, 0.5)
        scores: List[Tensor]: the scores of the extracted keypoints
            len(scores) = B   scores.shape = Nb
    """
    if score_map.ndim == 4:
        assert score_map.shape[1] == 1
        score_map = score_map.clone().squeeze(1)
    assert nms_radius % 2 == 1, "nms_radius must be odd"

    # if there are nans in the border of the score map, the border of the max_pool+kernel_size//2 will be nan
    # this thing make sense as if the max was really in the edge before the nan, we can't anyway trust it
    max_pool = F.max_pool2d(
        score_map[:, None, :, :],
        kernel_size=nms_radius,
        stride=1,
        padding=nms_radius // 2,
    ).squeeze(
        1
    )  # BxHxW
    maxima_mask = score_map == max_pool  # BxHxW

    # remove border detections
    if border == 0:
        map_masked = score_map
    else:
        mask = torch.zeros_like(score_map, dtype=torch.bool)  # BxHxW
        mask[:, border:-border, border:-border] = True
        map_masked = score_map * mask

    maxima_mask *= map_masked >= thr  # BxHxW

    if edge_thr > 0:
        corner_mask = compute_corner_mask(score_map, dilation=1, edge_thr=edge_thr)
        maxima_mask *= corner_mask

    B = score_map.shape[0]
    kpts_list = []
    scores_list = []
    for b in range(B):
        # transpose the map before the nonzero such that the coordinates are given in (x, y)
        kpts = torch.nonzero(maxima_mask[b].permute(1, 0))  # Nx2  (x,y)
        kpts_scores = score_map[b][kpts[:, 1], kpts[:, 0]]  # N

        # if there are too many keypoints, keep only the ones with higher score, in any case sort them by score
        top_kpts_idxs = torch.topk(kpts_scores, min(max_kpts, kpts.shape[0]))[1]
        kpts = kpts[top_kpts_idxs]
        kpts_scores = kpts_scores[top_kpts_idxs]

        # add +0.5 to the point to match the convention top-left center = (0.5, 0.5)
        kpts = kpts.float() + 0.5

        kpts_list.append(kpts)
        scores_list.append(kpts_scores)

    return kpts_list, scores_list


def generate_round_kernel_indices(kernel_radius: int) -> Tensor:
    """generate the indices for a round mask centered in 0 and with the specified kernel_radius
    Args:
        kernel_radius: the mask radius
    Returns:
        mask_idxs: the indices (y, x) of the mask
    """
    mask_idxs = []
    for i in range(-kernel_radius, kernel_radius + 1):
        for j in range(-kernel_radius, kernel_radius + 1):
            r = i**2 + j**2
            if r <= kernel_radius**2:
                mask_idxs.append((j, i))
    mask_idxs = torch.tensor(mask_idxs, dtype=torch.long).view(
        -1, 2
    )  # mask_area x 2     (y, x)

    return mask_idxs
