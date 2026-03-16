"""Functions that works with cameras, depthmaps and 3D points"""

from typing import Tuple, Union, Optional

import cv2
from torch import Tensor
import torch
import numpy as np

from utils.utils_2D import grid_sample_nan
from utils.utils_homography import (
    rotate_image_and_crop_without_black_borders,
    rot_mat,
)


def P_from_R_t(R: Tensor, t: Tensor) -> Tensor:
    """compose the P matrix from R and t
    Args:
        R: the rotation matrix
            Bx3x3
        t: the translation vector
            Bx3
    Return:
        P: the composed P matrix
            Bx4x4
    Raises:
        None
    """
    assert (
        R.ndim == 3 and t.ndim == 2
    ), f"Expected R to be Bx3x3 and t to be Bx3, got R={R.shape} and t={t.shape}"
    B = R.shape[0]
    P = torch.eye(4, device=R.device)[None, ...].repeat(B, 1, 1)
    P[:, :3, :3] = R
    P[:, :3, 3] = t
    return P


def P_from_R_t_np(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """compose the P matrix from R and t
    Args:
        R: the rotation matrix
            3x3
        t: the translation vector
            3
    Return:
        P: the composed P matrix
            4x4
    Raises:
        None
    """
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P


def invert_P(P: Tensor) -> Tensor:
    """invert the extrinsics P matrix in a more stable way with respect to np.linalg.inv()
    Args:
        P: input extrinsics P matrix
            Bx4x4
    Return:
        P_inv: the inverse of the P matrix
            Bx4x4
    Raises:
        None
    """
    B = P.shape[0]
    R = P[:, 0:3, 0:3]
    t = P[:, 0:3, 3:4]
    P_inv = torch.cat((R.permute(0, 2, 1), -R.permute(0, 2, 1) @ t), dim=2)
    P_inv = torch.cat(
        (P_inv, P.new_tensor([[0.0, 0.0, 0.0, 1.0]])[None, ...].repeat(B, 1, 1)), dim=1
    )
    return P_inv


def to_homogeneous(xy: Tensor) -> Tensor:
    return torch.cat((xy, torch.ones_like(xy[..., 0:1])), dim=-1)


def from_homogeneous(points: Tensor, eps: float = 1e-8) -> Tensor:
    z_vec: Tensor = points[..., -1:]
    # set the results of division by zero/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))
    output = scale * points[..., :-1]
    return output


def unproject_to_virtual_plane(
    xy: Tensor, K: Tensor, cast_to_double: bool = True
) -> Tensor:
    """unproject points to the camera virtual plane at depth 1
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            Bxnx2
        K: intrinsics of the camera
            Bx3x3
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns:
        xyz: 3D points laying on the virtual plane
            Bxnx3
    """
    xy_hom = to_homogeneous(xy)  # Bxnx3
    if cast_to_double:
        original_type = xy.dtype
        # Bx3x3 * Bx3xn = Bx3xn  -> Bxnx3 after permute
        xyz = (
            (
                torch.inverse(K.to(torch.double))
                @ (xy_hom.permute(0, 2, 1).to(torch.double))
            )
            .permute(0, 2, 1)
            .to(original_type)
        )
    else:
        # Bx3x3 * Bx3xn = Bx3xn  -> Bxnx3 after permute
        xyz = (torch.inverse(K) @ (xy_hom.permute(0, 2, 1))).permute(0, 2, 1)

    return xyz


def unproject_to_3D(xy: Tensor, K: Tensor, depths: Tensor) -> Tensor:
    """unproject points to 3D in the camera ref system
    Args:
        xy: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            Bxnx2
        K: intrinsics of the camera
            Bx3x3
        depths: the points depth
            Bxn
    Returns:
        xyz: unprojected 3D points in the camera reference system
            Bxnx3
    """
    assert xy.shape[0] == K.shape[0] and xy.shape[0] == depths.shape[0]
    assert xy.shape[1] == depths.shape[1]
    assert xy.shape[2] == 2

    xyz = unproject_to_virtual_plane(xy, K)  # Bxnx3
    xyz *= depths[:, :, None]  # Bxnx3

    return xyz


def change_reference_3D_points(
    xyz0: Tensor, P0: Tensor, P1: Tensor, cast_to_double: bool = True
) -> Tensor:
    """move 3D points from P0 to P1 reference systems
    Args:
        xyz0: the 3D points in the P0 coordinate system
            Bxnx3
        P0: the source coordinate system
            Bx4x4
        P1: the destination coordinate system
            Bx4x4
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns
        xyz1: the 3D points in the P1 coordinate system
            Bxnx3
    """
    xyz0_hom = to_homogeneous(xyz0)  # Bxnx4
    if cast_to_double:
        original_dtype = xyz0.dtype
        P0_inv = invert_P(P0.to(torch.double))
        xyz1_hom = (
            P1.to(torch.double) @ P0_inv @ xyz0_hom.permute(0, 2, 1).to(torch.double)
        )  # Bx4xn
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1)).to(original_dtype)  # Bxnx3
    else:
        P0_inv = invert_P(P0)
        xyz1_hom = P1 @ P0_inv @ xyz0_hom.permute(0, 2, 1)  # Bx4xn
        xyz1 = from_homogeneous(xyz1_hom.permute(0, 2, 1))  # Bxnx3

    return xyz1


def project_to_2D(
    xyz: Tensor,
    K: Tensor,
    img_shape: Tuple[int, int] = None,
    cast_to_double: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """project 3D points to 2D using the provided intrinsics matrix K. If img_shape is provided, set to nan the points
    that project out of the img and additionally return mask_outside boolean tensor
    Args:
        xyz: the 3D points
            Bxnx3
        K: the camera intrinsics matrix
            Bx3x3
        img_shape: if provided, set to nan the points that map out of the image and additionally return mask_outside
        cast_to_double: if true, cast to double before computation and cast back to the original type afterward
    Returns
        xy_proj: the 2D projection of the 3D points
            Bxnx2
        mask_outside: optional (if img_shape is provided). True where the point map outside img_shape
            Bxn bool
    """
    if cast_to_double:
        original_dtype = xyz.dtype
        # Bx3x3 * Bx3xn =  Bx3xn  -> Bxnx3 after permutation
        xy_proj_hom = (
            K.to(torch.double) @ xyz.permute(0, 2, 1).to(torch.double)
        ).permute(0, 2, 1)
        xy_proj = from_homogeneous(xy_proj_hom).to(original_dtype)  # Bxnx2
    else:
        xy_proj_hom = (K @ xyz.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # Bx3x3 * Bx3xn =  Bx3xn  -> Bxnx3 after permutation
        xy_proj = from_homogeneous(xy_proj_hom)  # Bxnx2

    if img_shape is not None:
        # filter points that fall outside the second image but have depth valid
        # as the comparison of a 'nan' values with something else is always false, only the points that had valid
        # depth will appear in mask_outside
        mask_outside = (
            (xy_proj[..., 0] < 0)
            + (xy_proj[..., 0] > img_shape[1])
            + (xy_proj[..., 1] < 0)
            + (xy_proj[..., 1] > img_shape[0])
        )
        xy_proj[mask_outside] = float("nan")
        return xy_proj, mask_outside
    else:
        return xy_proj


def reproject_2D_2D(
    xy0: Tensor,
    depthmap0: Tensor,
    P0: Tensor,
    P1: Tensor,
    K0: Tensor,
    K1: Tensor,
    img1_shape: Tuple[int, int] = None,
    mode: str = "nearest",
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """returns projected xy0 points from img0 to img1 using depth0. Points that have an invalid depth='nan' are
    set to 'nan' (if bilinear sampling is used, all the 4 closest depth values must be valid to get a valid projection).
    If img1_shape is provided, also the points that project out of the second image are set to Nan
    Args:
        xy0: xy points in img0 (with convention top-left pixel coordinate (0.5, 0.5)
            Bxnx2
        depthmap0: depthmap of img0
            BxHxW
        P0: camera0 extrinsics matrix
            Bx4x4
        P1: camera1 extrinsics matrix
            Bx4x4
        K0: camera0 intrinsics matrix
            Bx3x3
        K1: camera1 intrinsics matrix
            Bx3x3
        img1_shape: shape of img1 (H, W)
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
    Returns:
        xy0_proj: the projected keypoints in img1
            Bxnx2
        mask_invalid_depth: mask of points that had invalid depth
            Bxn  bool
        mask_outside: optional (if img1_shape is provided) mask of points that had valid depth but project out of the
            second image
            Bxn  bool
    """
    # interpolate depths
    selected_depths0, mask_invalid_depth0 = grid_sample_nan(
        xy0, depthmap0, mode=mode
    )  # Bxn, Bxn

    # use the depth to define the 3D coordinates of points in the ref system of camera0
    xyz0 = unproject_to_3D(xy0, K0, selected_depths0)  # Bxnx3

    # change the ref system of the 3d point to camera1
    xyz0_proj = change_reference_3D_points(xyz0, P0, P1)  # Bxnx3

    # project the point in the destination image
    if img1_shape is not None:
        xy0_proj, mask_outside0 = project_to_2D(
            xyz0_proj, K1, img1_shape
        )  # Bxnx2, Bxnx2
        return xy0_proj, mask_invalid_depth0, mask_outside0
    else:
        xy0_proj = project_to_2D(xyz0_proj, K1)  # Bxnx2, Bxnx2
        return xy0_proj, mask_invalid_depth0


def depth_consistency_check(
    xy0_proj: Tensor,
    selected_depths0: Tensor,
    depthmap1: Tensor,
    P0: Tensor,
    P1: Tensor,
    K1: Tensor,
    max_relative_depth_error: float = 0.1,
    mode="nearest",
) -> Tuple[Tensor, Tensor, Tensor]:
    """check if the depth extracted from the projected locations xy0_proj is consistent with the source ones
    Args:
        xy0_proj: xy0 points in img0 projected in img1 (with convention top-left pixel coordinate (0.5, 0.5). The depth
            is sampled at these locations to check if it consistent with the source one
            Bxnx2
        selected_depths0: the depth of the source xy0 points
            Bxn
        depthmap1: depthmap of img0
            BxHxW
        P0: camera0 extrinsics matrix
            Bx4x4
        P1: camera1 extrinsics matrix
            Bx4x4
        K1: camera1 intrinsics matrix
            Bx3x3
        max_relative_depth_error: the max relative depth error for a point to be considered valid. Error computed as
            (depth0 - depth1) / min(depth0, depth1) where depth1 is the depth of the projected point xy0_proj
            interpolated from depth1, projected back in 3D space and in the reference frame of depth0
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
    Returns:
        mask_inconsistent_depth: true where the depth is inconsistent
            Bxn
        mask_invalid_depth1: true where depth1 was invalid
            Bxn
        xyz1_proj: the 3D coordinates in img0 coordinate frame of the backprojected points xy0_proj using depth1 with
            nan where the depth is inconsistent
            Bxnx3
    """
    # interpolate depths in depth1
    selected_depths1, mask_invalid_depth1 = grid_sample_nan(
        xy0_proj, depthmap1, mode=mode
    )  # Bxn, Bxn
    xyz1 = unproject_to_3D(xy0_proj, K1, selected_depths1)  # Bxnx3

    # change the ref system of the 3d point to camera1
    xyz1_proj = change_reference_3D_points(xyz1, P1, P0)

    # check for depth inconsistencies
    relative_depth_error = torch.abs(selected_depths0 - xyz1_proj[:, :, 2]) / torch.min(
        selected_depths0, xyz1_proj[:, :, 2]
    )  # Bxn
    # we check with > because in case of nan in relative_depth_error (coming from xyz0 or xyz1_proj) we don't want
    # this to be marked as inconsistent depth
    mask_inconsistent_depth = relative_depth_error > max_relative_depth_error  # Bxn
    xyz1_proj[mask_inconsistent_depth[:, :, None].expand(-1, -1, 3)] = float(
        "nan"
    )  # Bxn

    return mask_inconsistent_depth, mask_invalid_depth1, xyz1_proj


def compute_GT_matches_matrix_3D(
    xy0: Tensor,
    xy1: Tensor,
    depthmap0: Tensor,
    depthmap1: Tensor,
    P0: Tensor,
    P1: Tensor,
    K0: Tensor,
    K1: Tensor,
    max_relative_depth_error: float = 0.1,
    max_pixel_error: float = 3.0,
    min_pixel_error_for_unmatched: float = 5.0,
    mode: str = "nearest",
    return_distances_and_projected: bool = False,
    allow_multiple_matches: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """computes the GT matching matrix from the provided data. The resulting matrix will have dimension Bx(n0+1)x(n1+1)
    - a match is considered INLIER if all the following conditions are True:
        - depth of xy0 is not 'nan' and xy0_proj ends inside img1
        - depth of xy1 is not 'nan' and xy1_proj ends inside img0
        - dist0 (distance between xy0 and xy1_proj) is minimum along columns and < max_pixel_error
        - dist1 (distance between xy0_proj and xy1) is minimum along rows and < max_pixel_error
        A True is set in the output matching matrix
    - a keypoint xy0 is considered UNMATCHED if any of the following conditions are True (similarly for xy1):
        - depth of xy0 is valid and xy0_proj ends outside img1
        - depth of xy0 is valid and xy0_proj ends inside img1, there are no xy1 closer than min_pixel_for_unmatched
        - depth of xy0 is valid and xy0_proj ends inside img1, there is only one xy1 closer than
            min_pixel_for_unmatched, it has depth but project out of img0
        - depth of xy0 is valid and xy0_proj ends inside img1, there is only one xy1 closer than
            min_pixel_for_unmatched, it has depth and project inside img0, xy1_proj is not closer than
            min_pixel_for_unmatched to the original xy0
        A True is set in the output matching matrix bin
    - all other cases are "unsure", rows or columns will be completely filled with 0s
    Note: keypoints with invalid depth does not appear in the matching matrix
    Args:
        xy0: xy points in img0
            Bxn0x2
        xy1: xy points in img1
            Bxn1x2
        depthmap0: depthmap of img0
            BxHxW
        depthmap1: depthmap of img1
            BxHxW
        P0: camera0 extrinsics matrix
            Bx4x4
        P1: camera1 extrinsics matrix
            Bx4x4
        K0: camera0 intrinsics matrix
            Bx3x3
        K1: camera1 intrinsics matrix
            Bx3x3
        max_relative_depth_error: the max relative depth error for a point to be considered valid. Error computed as
            (depth0 - depth1) / min(depth0, depth1) where depth1 is the depth of the projected point xy0_proj
            interpolated from depth1, projected back in 3D space and in the reference frame of depth0
        max_pixel_error: the max symmetric reproj error to be considered a match
        min_pixel_error_for_unmatched: if two keypoints are mnn, max(dist0, dist1) must be greater than this to be
            considered unmatched
        mode: depthmap interpolation mode, can be 'nearest' or 'bilinear'
        return_distances_and_projected: additionally return computed distances
        allow_multiple_matches: if true, we don't require the keypoints to be mutual nearest neighbor, but any pair
            of keypoints that projects in both images close enough is considered a match. This is useful in the case
            of multiscale extraction, where the same point (or almost the same) can be extracted at multiple scales
    Returns:
        matching_matrix: the output matching matrix with bins for the unsure
            B x n0+1 x n1+1  bool
    """
    device = xy0.device
    B, n0, n1 = xy0.shape[0], xy0.shape[1], xy1.shape[1]

    # ==================== img0 -> img1 ==============================================================================
    # interpolate depths
    selected_depths0, mask_invalid_depth0 = grid_sample_nan(
        xy0, depthmap0, mode=mode
    )  # Bxn, Bxn
    # use the depth to define the 3D coordinates of points in the ref system of camera0
    xyz0 = unproject_to_3D(xy0, K0, selected_depths0)  # Bxnx3
    # change the ref system of the 3d point to camera1
    xyz0_proj = change_reference_3D_points(xyz0, P0, P1)  # Bxnx3
    # project the point in the destination image. The index 1 of mask_outside1 refers to the fact that this is
    # computed in img1 space
    xy0_proj, mask_outside1 = project_to_2D(
        xyz0_proj, K1, depthmap1.shape[1:]
    )  # Bxnx2, Bxnx2
    # check for depth consistency and set xy0_proj to nan if inconsistent
    mask_inconsistent_depth0 = depth_consistency_check(
        xy0_proj,
        selected_depths0,
        depthmap1,
        P0,
        P1,
        K1,
        max_relative_depth_error=max_relative_depth_error,
        mode=mode,
    )[
        0
    ]  # Bxn0
    xy0_proj[mask_inconsistent_depth0] = float("nan")
    # ==================== img1 -> img0 ==============================================================================
    # interpolate depths
    selected_depths1, mask_invalid_depth1 = grid_sample_nan(
        xy1, depthmap1, mode=mode
    )  # Bxn, Bxn
    # use the depth to define the 3D coordinates of points in the ref system of camera1
    xyz1 = unproject_to_3D(xy1, K1, selected_depths1)  # Bxnx3
    # change the ref system of the 3d point to camera0
    xyz1_proj = change_reference_3D_points(xyz1, P1, P0)  # Bxnx3
    # project the point in the destination image. The index 0 of mask_outside0 refers to the fact that this is
    # computed in img0 space
    xy1_proj, mask_outside0 = project_to_2D(
        xyz1_proj, K0, depthmap0.shape[1:]
    )  # Bxnx2, Bxnx2
    # check for depth consistency and set xy1_proj to nan if inconsistent
    mask_inconsistent_depth1 = depth_consistency_check(
        xy1_proj,
        selected_depths1,
        depthmap0,
        P1,
        P0,
        K0,
        max_relative_depth_error=max_relative_depth_error,
        mode=mode,
    )[
        0
    ]  # Bxn1
    xy1_proj[mask_inconsistent_depth1] = float("nan")

    # compute the inconsistent depth overall map
    mask_inconsistent_depth = (
        mask_inconsistent_depth0[:, :, None] + mask_inconsistent_depth1[:, None, :]
    )  # Bxn0xn1

    # compute the distances in both images
    dist0 = torch.cdist(
        xy0, xy1_proj, compute_mode="donot_use_mm_for_euclid_dist"
    )  # Bxn0xn1
    dist1 = torch.cdist(
        xy0_proj, xy1, compute_mode="donot_use_mm_for_euclid_dist"
    )  # Bxn0xn1

    # set to inf all the distances that result nan
    dist0[dist0.isnan()] = float("+inf")
    dist1[dist1.isnan()] = float("+inf")

    # True where a pair projected-detected is really close
    in_radius_small0 = dist1 < max_pixel_error  # Bxn0xn1
    in_radius_small1 = dist0 < max_pixel_error  # Bxn0xn1
    # True if the two pairs are inside the small radius in both images
    in_radius_small = in_radius_small0 * in_radius_small1  # Bxn0xn1

    # True where a pair projected-detected are withing the big threshold
    in_radius_big0 = dist1 < min_pixel_error_for_unmatched  # Bxn0xn1
    in_radius_big1 = dist0 < min_pixel_error_for_unmatched  # Bxn0xn1
    # True for the entire columns (xy0 proj in img1) or the entire row (xy1 proj in img0) if there is only one
    # neighbour within the big threshold
    single_ngh0 = in_radius_big0.to(torch.int).sum(2, keepdim=True) == 1  # Bxn0xn1
    single_ngh1 = in_radius_big1.to(torch.int).sum(1, keepdim=True) == 1  # Bxn0xn1

    nn0 = torch.zeros(B, n0, n1, dtype=torch.bool, device=device)  # Bxn0xn1
    nn1 = torch.zeros(B, n0, n1, dtype=torch.bool, device=device)  # Bxn0xn1
    nn0_idx = dist1.min(2, keepdim=True)[1]  # Bxn0x1
    nn1_idx = dist0.min(1, keepdim=True)[1]  # Bx1xn1
    # True for the nearest neighbour of every projected point
    nn0.scatter_(2, nn0_idx, torch.ones_like(nn0, dtype=torch.bool))  # Bxn0xn1
    nn1.scatter_(1, nn1_idx, torch.ones_like(nn1, dtype=torch.bool))  # Bxn0xn1
    # True if the points is both nearest neighbour and it's the only one within the big threshold
    single_nn0 = nn0 * single_ngh0  # Bxn0xn1
    single_nn1 = nn1 * single_ngh1  # Bxn0xn1
    mnn_mask = nn0 * nn1  # Bxn0xn1

    # This works the best
    if allow_multiple_matches:
        gt_matches = in_radius_small
    else:
        gt_matches = mnn_mask * in_radius_small
    # This is wrong because finds multiple matches for the same
    # gt_matches = single_nn0 * in_radius_small1 + single_nn1 * in_radius_small0
    # This is too much restrictive
    # gt_matches = single_nn0 * single_nn1 * in_radius_small

    # point have depth and project out of the other image image
    bin0 = mask_outside1
    bin1 = mask_outside0

    # point have depth and the depth is consistent, project inside the other image.
    # There are no point in the other image within 5 px from the projected one
    bin0 += ~mask_invalid_depth0 * ~mask_inconsistent_depth0 * ~(in_radius_big0.any(2))
    bin1 += ~mask_invalid_depth1 * ~mask_inconsistent_depth1 * ~(in_radius_big1.any(1))

    # point have depth and the depth is consistent, project inside the other image. There is only one point in the
    # other image within 5 px.It has consistent depth and project out of the first image
    bin0 += (single_nn0 * ~mask_inconsistent_depth * mask_outside0[:, None, :]).any(2)
    bin1 += (single_nn1 * ~mask_inconsistent_depth * mask_outside1[:, :, None]).any(1)

    # point have depth and is consistent, project inside the other image. There is only one point in the other image
    # within 5 px. It has depth and project inside the first image, but not within 5px from the original point
    bin0 += (
        single_nn0
        * ~in_radius_big1
        * ~mask_invalid_depth1[:, None, :]
        * ~mask_inconsistent_depth
    ).any(2)
    bin1 += (
        single_nn1
        * ~in_radius_big0
        * ~mask_invalid_depth0[:, :, None]
        * ~mask_inconsistent_depth
    ).any(1)

    # building the matching matrix composing gt_matches and bins
    matching_matrix = torch.zeros(B, n0 + 1, n1 + 1, device=device, dtype=torch.bool)
    matching_matrix[:, :n0, :n1] = gt_matches
    matching_matrix[:, :n0, -1] = bin0
    matching_matrix[:, -1, :n1] = bin1
    # print('THIS MUST BE 1', matching_matrix[:, :, :-1].sum(1).max())
    # print('THIS MUST BE 1', matching_matrix[:, :-1, :].sum(2).max())

    if return_distances_and_projected:
        return matching_matrix, xy0_proj, xy1_proj, dist0, dist1

    return matching_matrix


def scale_and_crop(
    img: np.ndarray,
    K0: np.ndarray,
    output_shape: Union[Tuple[int, int], np.ndarray],
    depth: Optional[np.ndarray] = None,
    center: Optional[np.ndarray] = None,
    max_random_offset: int = 0,
    allow_scaling: bool = False,
):
    assert img.ndim in {2, 3}, f"img.ndim {img.ndim} must be 2 or 3"
    assert K0.ndim == 2, f"K0.ndim {K0.ndim} must be 2"
    assert K0.shape == (3, 3), f"K0.shape {K0.shape} must be (3, 3)"
    assert (
        img.shape[:2] == depth.shape
    ), f"img.shape[:2] {img.shape[:2]} must be equal to depth.shape {depth.shape}"

    H, W = img.shape[:2]
    H_out, W_out = output_shape
    K0_out = np.copy(K0)

    # check if the crop fit in the image, otherwise upscale the image
    if (H_out / H) > 1 or (H_out / W) > 1:
        if not allow_scaling:
            raise ValueError(
                "The crop is bigger than the image and scaling is not allowed"
            )

    if center is not None:
        # check if the crop fit in the image, otherwise upscale the image
        if (H_out / H) > 1 or (H_out / W) > 1:
            scale_factor = max(H_out / H, H_out / W)
            img = cv2.resize(
                img,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA,
            )
            H, W = img.shape[:2]
            # correct the intrinsics scale
            K0_out[:2, :] *= scale_factor
            if depth is not None:
                inv_depth = 1 / depth
                inv_depth = cv2.resize(
                    inv_depth,
                    None,
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
                depth = 1 / inv_depth
        else:
            scale_factor = 1

        # crop the image so that it conform to the output_shape. The cropping is centered in the center
        # coordinates with a random offset
        top_left_centered = center - np.array([W_out, H_out]) // 2

        if max_random_offset > 0:
            offset_x = np.random.randint(
                -min(max_random_offset, top_left_centered[0]),
                min(max_random_offset, W - top_left_centered[0] - W_out),
            )
            offset_y = np.random.randint(
                -min(max_random_offset, top_left_centered[1]),
                min(max_random_offset, H - top_left_centered[1] - H_out),
            )
            offset = np.array([offset_x, offset_y])
        else:
            offset = np.array([0, 0])
        top_left = top_left_centered + offset
    else:
        raise NotImplemented

    bbox = np.array(
        [[top_left[1], top_left[0]], [top_left[1] + H_out, top_left[0] + W_out]]
    )  # [y0, x0], [y1, x1]

    img_out = img[bbox[0, 0] : bbox[1, 0], bbox[0, 1] : bbox[1, 1]]
    if depth is not None:
        depth_out = depth[bbox[0, 0] : bbox[1, 0], bbox[0, 1] : bbox[1, 1]]

    assert img_out.shape[:2] == (
        H_out,
        W_out,
    ), f"img.shape[:2] {img.shape[:2]} must be equal to (H_out, W_out) {(H_out, W_out)}"

    # correct the intrinsics translation
    K0_out[0, 2] -= top_left[0]
    K0_out[1, 2] -= top_left[1]

    if depth is not None:
        return img_out, K0_out, scale_factor, bbox, depth_out
    else:
        return img_out, K0_out, scale_factor, bbox


def rotate_image_and_camera_z_axis(
    angle_degrees: float,
    img: Tensor,
    P: Tensor,
    K: Tensor,
    depth: Optional[Tensor] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    assert P.ndim == 2, f"P.ndim {P.ndim} must be 2"

    if (2 * K[0, 2]).round().int() != img.shape[-1] or (
        2 * K[1, 2]
    ).round().int() != img.shape[-2]:
        print(
            "WARNING: img center is not centered with the intrinsics, the rotation will be wrong",
            f"found img shape {img.shape}, K[0, 2] {K[0, 2]}, K[1, 2] {K[1, 2]}",
        )

    img_rotated, hom, left_top = rotate_image_and_crop_without_black_borders(
        img, angle_degrees
    )

    # correct camera
    K_rotated = K.clone()
    K_rotated[0, 2] -= left_top[0]
    K_rotated[1, 2] -= left_top[1]

    # apply a rotation around the Z axis
    R, T = P[:3, :3].to(torch.double), P[:3, 3].to(torch.double)
    R_rotation = rot_mat(angle_degrees / 180.0 * np.pi, dtype=torch.float64)
    R_composed = R_rotation @ R
    # # the following lines are equivalent
    # center = -R.inverse() @ T
    # T_composed = -R_composed @ center
    T_composed = R_rotation @ T
    P_rotated = P_from_R_t(R_composed[None], T_composed[None])[0]

    if depth is not None:
        depth_rotated = rotate_image_and_crop_without_black_borders(
            depth[None], angle_degrees, mode="nearest"
        )[0][0]
        return img_rotated, P_rotated, K_rotated, depth_rotated
    else:
        return img_rotated, P_rotated, K_rotated

    # from libutils.utils_3D import test_GT_matches_extraction_3D
    # # test if the transformation we applied is correct
    # xy0 = torch.tensor([[500.0, 200.0],
    #                  [300.0, 200.0]])
    # xy1 = torch.zeros((1, 2))
    # test_GT_matches_extraction_3D(xy0[None], xy1[None], depth[None], depth_rotated[None], P[None], P_rotated[None],
    #                               K[None], K_rotated[None], img0=img[None], img1=img_rotated[None])
