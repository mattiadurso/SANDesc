"""homography utils"""

import math
import sys

sys.path.append("../")
from typing import Tuple, Dict, Union, Collection
import numpy as np
from numpy import array
import torch
from torch import Tensor
import torch.nn.functional as F
from utils.utils_2D import (
    filter_outside,
    mutual_nearest_neighbors_from_dist_matrices,
    compute_quadrilateral_area_from_corners,
    rotatedRectWithMaxArea,
)


def apply_with_probability(probability: float) -> bool:
    """return True or False with the specified probability
    Args:
        probability: the specified probability
    Returns:
        bool: true or false with the specified probability

    """
    return np.random.rand() < probability


def rot_mat(
    alpha: float, dtype: torch.dtype = None, device: Union[str, torch.device] = None
) -> Tensor:
    r_mat = torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    return r_mat


def transl_mat(
    xy: Union[Tensor, Tuple],
    dtype: torch.dtype = None,
    device: Union[str, torch.device] = None,
) -> Tensor:
    """

    Returns:
        object:
    """
    t_mat = torch.tensor(
        [[1.0, 0.0, xy[0]], [0.0, 1.0, xy[1]], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return t_mat


def transl_mat_numpy(xy: Union[np.ndarray, Tuple]) -> np.ndarray:
    t_mat = np.array([[1.0, 0.0, xy[0]], [0.0, 1.0, xy[1]], [0.0, 0.0, 1.0]])
    return t_mat


def rot_mat_numpy(alpha: float) -> np.ndarray:
    rot_mat = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot_mat


def scale_mat(scale: Union[float, Tensor, Tuple]) -> Tensor:
    if isinstance(scale, Collection):
        s = scale
    else:
        s = (scale, scale)
    scale_matrix = torch.tensor([[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 1.0]])
    return scale_matrix


def scale_mat_numpy(scale: Union[float, np.ndarray, Tuple]) -> np.ndarray:
    if isinstance(scale, Collection):
        s = scale
    else:
        s = (scale, scale)
    scale_mat = np.array([[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 1.0]])
    return scale_mat


def shear_mat_numpy(shear: Union[np.ndarray, Tuple]) -> np.ndarray:
    shear_mat = np.array([[1.0, shear[0], 0.0], [shear[1], 1.0, 0.0], [0.0, 0.0, 1.0]])
    return shear_mat


def perspective_mat_numpy(perspective: Union[np.ndarray, Tuple]) -> np.ndarray:
    perspective_mat = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective[0], perspective[1], 1.0]]
    )
    return perspective_mat


def warp_points(
    xy: Tensor,
    H: Tensor,
    img_shape: Union[np.ndarray, Tensor, Tuple[int, int]] = None,
    border: int = 0,
) -> Tensor:
    """warp the points using the provided homography matrix
    Args:
        xy: input points coordinates with order (x, y)
           B,n,2
        H: input homography
           B,3,3
        img_shape: if provided, the points that project out of shape are set to 'nan'
        border: together with img_shape, set to nan the points that are closer to the border than this
    Returns:
        Tensor: the projected points
           B,n,2
    """
    assert xy.shape[0] == H.shape[0]
    assert xy.shape[2] == 2
    assert H.shape[1] == 3 and H.shape[2] == 3
    if isinstance(img_shape, Tensor) or isinstance(img_shape, np.ndarray):
        assert img_shape.shape == (2,)

    # xy_hom = geom.convert_points_to_homogeneous(xy.to(H.dtype))  # B,n,3
    xy_hom = torch.cat(
        (
            xy,
            torch.ones((xy.shape[0], xy.shape[1], 1), dtype=xy.dtype, device=xy.device),
        ),
        dim=2,
    ).to(
        H.dtype
    )  # B,n,3
    xy_proj_hom = xy_hom @ H.to(xy.device).permute(0, 2, 1)  # B,n,3
    # xy_proj = geom.convert_points_from_homogeneous(xy_proj_hom)   # B
    xy_proj = xy_proj_hom[:, :, 0:2] / xy_proj_hom[:, :, 2:3]  # B,n,2

    if img_shape is not None:
        xy_proj = filter_outside(xy_proj, img_shape, border)

    xy_proj = xy_proj.to(xy.dtype)

    return xy_proj


def warp_points_numpy(points: array, H: array) -> array:
    """warp the points using the provided homography matrix
    Args:
        points: the input coordinates
            nx2
        H: input homography
            3x3
    Returns:
        array: the projected points
            nx2
    """
    points_dst = H.dot(np.concatenate((points.T, np.ones((1, points.shape[0])))))
    points_dst = points_dst[0:2, :] / points_dst[2, :]
    return points_dst.T


def get_dist_matrix(
    xy0: Tensor,
    xy1: Tensor,
    H0_1: Tensor,
    img1_shape: Tuple[int, int] = None,
    border: int = 0,
    return_projected: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """compute the distance matrix between the keypoints projected from img0 (rows) to img1 (column) using
    the provided homography
    Args:
        xy0: input points in img0
            Bxn0x2
        xy1: input points in img1
            Bxn1x2
        H0_1: homography that warps img0 in img1
            Bx3x3
        img1_shape: the shape of img1
        border: together with img1_shape, used to invalidate the points that are close to the image borders
        return_projected: additionally return the projected points
    Returns:
        Tensor: distance computed in img1 between the projected points xy0_proj and the points in img1
            CAN HAVE 'nan' if img_shape is provided and the points project out of the second image
            Bxn0xn1
    """
    xy0_proj = warp_points(xy0, H0_1, img1_shape, border)
    # compute the distance between each xy0_proj and xy1
    dist_matrix = torch.cdist(
        xy0_proj, xy1, compute_mode="donot_use_mm_for_euclid_dist"
    )

    if return_projected:
        return dist_matrix, xy0_proj

    return dist_matrix


def compute_GT_matches_matrix_homography(
    xy0: Tensor,
    xy1: Tensor,
    H0_1: Tensor,
    thr: float = 3.0,
    img0_shape: Tuple[int, int] = None,
    img1_shape: Tuple[int, int] = None,
    border: int = 0,
    return_distances_and_projected: bool = False,
) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
    """compute GT matching matrix using the homography that maps img0 in img1
    Args:
        xy0: input points in img0
            Bxn0x2
        xy1: input points in img1
            Bxn1x2
        H0_1: homography that warps img0 in img1
            Bx3x3
        thr: threshold for considering a match valid
        img0_shape: the shape of img0, points that project out are considered invalid
        img1_shape: the shape of img0, points that project out are considered invalid
        border: all the keypoints that are closer to the border than this value will not generate valid matches
        return_distances_and_projected: additionally return computed distances
    Returns:
        GT_matching_matrix: the resulting GT matching matrix
            B x n0+1 x n1+1   torch.bool
    """
    dist_matrix0, xy1_proj = get_dist_matrix(
        xy1, xy0, torch.inverse(H0_1), img0_shape, border, return_projected=True
    )
    dist_matrix0 = dist_matrix0.permute(0, 2, 1)
    dist_matrix1, xy0_proj = get_dist_matrix(
        xy0, xy1, H0_1, img1_shape, border, return_projected=True
    )
    dist_matrix0[dist_matrix0.isnan()] = float("+inf")
    dist_matrix1[dist_matrix1.isnan()] = float("+inf")

    mnn = mutual_nearest_neighbors_from_dist_matrices(dist_matrix0, dist_matrix1)
    B, n0, n1 = mnn.shape

    GT_matching_matrix_with_bins = mnn.new_zeros(B, n0 + 1, n1 + 1)
    GT_matching_matrix_with_bins[:, :-1, :-1] = (
        mnn * (dist_matrix0 < thr) * (dist_matrix1 < thr)
    )

    # fill the bins with everything that was not matched
    GT_matching_matrix_with_bins[:, :-1, -1] = ~GT_matching_matrix_with_bins[
        :, :-1, :-1
    ].any(2)
    GT_matching_matrix_with_bins[:, -1, :-1] = ~GT_matching_matrix_with_bins[
        :, :-1, :-1
    ].any(1)

    if return_distances_and_projected:
        return (
            GT_matching_matrix_with_bins,
            xy0_proj,
            xy1_proj,
            dist_matrix0,
            dist_matrix1,
        )

    return GT_matching_matrix_with_bins


def points_in_image(xy: np.ndarray, img_shape: Tuple[int, int]) -> bool:
    """check if all the points coordinate are contained inside the image
    Args:
        xy: input points coordinate with (x, y)
        img_shape: the image shape where all the points must be inside
    Returns:
        bool: True if all the points lie inside the image
    """
    if (xy[:, 0] > 0).all() and (xy[:, 0] < img_shape[1]).all():
        if (xy[:, 1] > 0).all() and (xy[:, 1] < img_shape[0]).all():
            return True
    return False


def is_convex(xy: np.ndarray):
    """check if the polygon is convex"""
    N = xy.shape[0]

    # direction of cross product of previous edges
    direction_old = 0

    for i in range(N):
        # compute two consecutive edges
        vector = np.array([xy[(i + 1) % N], xy[(i + 2) % N]]) - xy[i]

        direction_new = np.cross(vector[0], vector[1])
        # if direction of cross product of all adjacent edges are not same
        if direction_new * direction_old < 0:
            return False
        else:
            # update old
            direction_old = direction_new
    return True


def generate_homography_for_patch_augmentation(
    patch_shape: Union[np.ndarray, Tensor],
    patch_center: Union[np.ndarray, Tensor],
    alpha_degrees: float = 0.0,
    x_shear: float = 0.0,
    y_shear: float = 0.0,
    x_translation: float = 0.0,
    y_translation: float = 0.0,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y_perspective: float = 0.0,
    x_perspective: float = 0.0,
) -> np.ndarray:
    assert len(patch_shape) == 2
    Hs = {}

    # SHEAR
    if x_shear != 0 or y_shear != 0:
        H_shear = shear_mat_numpy((x_shear, y_shear))
        Hs["shear"] = H_shear

    # PERSPECTIVE
    if x_perspective != 0 or y_perspective != 0:
        H_p = perspective_mat_numpy((x_perspective, y_perspective))
        # here we apply some corrections such that the transformation is more intuitive
        # correct for the translation
        patch_corners = np.array(
            [
                [0, 0],
                [0, patch_shape[0]],
                [patch_shape[1], patch_shape[0]],
                [patch_shape[1], 0],
            ]
        )
        patch_corners = patch_corners - np.flipud(patch_shape) / 2
        patch_corners_projected = warp_points_numpy(patch_corners, H_p)
        center = patch_corners_projected.mean(0)
        H_p = transl_mat_numpy(-center) @ H_p
        # correct for the scale
        area = compute_quadrilateral_area_from_corners(patch_corners_projected)
        scale = math.sqrt((patch_shape[0] * patch_shape[1]) / area)
        H_p = scale_mat_numpy(scale) @ H_p
        Hs["perspective"] = H_p

    # ROTATION
    if alpha_degrees != 0:
        H_rot = rot_mat_numpy(alpha_degrees / 180.0 * math.pi)
        H_r = H_rot
        Hs["rotation"] = H_r

    # SCALE
    if x_scale != 1 or y_scale != 1:
        H_s = scale_mat_numpy((x_scale, y_scale))
        Hs["scale"] = H_s

    # TRANSLATION
    if x_translation != 0 or y_translation != 0:
        H_t = transl_mat_numpy((x_translation, y_translation))
        Hs["translation"] = H_t

    keys = list(Hs.keys())
    H = np.eye(3)
    for k in keys:
        H = Hs[k] @ H

    # translate back and forward such that all the transformations are applyed at the center of the patch
    t = np.array([[1, 0, patch_center[0]], [0, 1, patch_center[1]], [0, 0, 1]])
    H = t @ H @ np.linalg.inv(t)

    # correct the homography such that when warping the image, the patch_center coordinate
    # ends exactly in the center of the patch
    translation = transl_mat_numpy(
        (patch_center[0] - patch_shape[1] // 2, patch_center[1] - patch_shape[0] // 2)
    )
    H = H @ translation

    return H


def sample_homography(
    patch_center: np.ndarray,
    patch_shape: Union[Tuple[int, int], np.ndarray],
    source_img_shape: Union[Tuple[int, int], np.ndarray],
    params: Dict[str, float],
    max_n_iterations: int = 1000,
) -> Tuple[np.ndarray, Dict]:
    """sample a random homography using the provided parameters
    Args:
        patch_center: the coordinates of the patch center given as (x, y)
        patch_shape: the patch shape that must completely fall inside the warped image given as (H, W)
        source_img_shape: the input image shape as (H, W)
        params: a dict that must contain
            'angle_delta': rotation standard deviation in degrees
            'angle_std': rotation standard deviation in degrees
            'angle_p': probability of applying rotation
            'translation_std': translation standard deviation given in relation to the patch_shape
                (1.0 means 100% of the patch dimension)
            'translation_delta': translation max and min value for uniform sampling given in relation to the patch_shape
                (1.0 means 100% of the patch dimension)
            'translation_p': probability of applying translation
            'shear_std': shear standard deviation
            'shear_delta': shear max and min value for uniform sampling
                (reasonable values are up to 2.0)
            'shear_p': probability of applying shear
                (reasonable values are up to 2.0)
            'scale_std': scale standard deviation
            'scale_delta': scale max and min value for uniform sampling
            'scale_p': probability of applying scale
            'scale_anisotropic': if true scale for x and y can be different
            'perspective_std': perspective standard deviation. The code corrects for scale and translation.
                (reasonable values are up to 0.5, max value is abs(perspective_x) + abs(perspective_y) = 2.0)
            'perspective_delta': perspective min and max for uniform sampling. The code corrects for scale and translation.
                (reasonable values are up to 0.5, max value is abs(perspective_x) + abs(perspective_y) = 2.0)
            'perspective_p': probability of applying perspective
            'allow_results_with_padding': if true enable homographies that would include outside of the image areas
                in the output patch
        max_n_iterations: the number of iterations before exiting returning the homography with just the translation
    Returns:
        H: sampled homography that maps from OUTPUT to INPUT
            3x3
        H_params: dict of selected parameters for the homography
    """
    assert ("angle_std" in params) != (
        "angle_delta" in params
    ), "only std or delta must be provided"
    assert ("translation_std" in params) != (
        "translation_delta" in params
    ), "only std or delta must be provided"
    assert ("shear_std" in params) != (
        "shear_delta" in params
    ), "only std or delta must be provided"
    assert ("scale_std" in params) != (
        "scale_delta" in params
    ), "only std or delta must be provided"
    assert ("perspective_std" in params) != (
        "perspective_delta" in params
    ), "only std or delta must be provided"

    H_params = {}

    for i in range(max_n_iterations):
        if (
            i == max_n_iterations - 1
        ):  # if we reached the max number of iterations return just the translation
            print(
                "Warning: we could not generate the homography, returning the homography with just the translation"
            )
            H = transl_mat_numpy(
                (
                    patch_center[0] - patch_shape[1] // 2,
                    patch_center[1] - patch_shape[0] // 2,
                )
            )
            H_params = {
                "translation": (
                    patch_center[0] - patch_shape[1] // 2,
                    patch_center[1] - patch_shape[0] // 2,
                )
            }
            break

        # SHEAR
        if apply_with_probability(params["shear_p"]) and (
            params.get("shear_std", 0) > 0
            or params.get("shear_delta", 0) > 0
            or params.get("shear_x_delta", 0) > 0
            or params.get("shear_y_delta", 0) > 0
        ):
            if "shear_std" in params:
                x_shear = np.random.normal(0, params["shear_std"])
                y_shear = np.random.normal(0, params["shear_std"])
            else:
                assert (
                    "shear_delta" in params
                    and "shear_x_delta" in params
                    and "shear_y_delta" in params
                    and "shear_anisotropic_ratio" in params
                )
                if np.random.random() < params["shear_anisotropic_ratio"]:
                    x_shear = (np.random.random() * 2 - 1) * params["shear_x_delta"]
                    y_shear = (np.random.random() * 2 - 1) * params["shear_y_delta"]
                else:
                    x_shear = (np.random.random() * 2 - 1) * params["shear_delta"]
                    y_shear = (np.random.random() * 2 - 1) * params["shear_delta"]
            H_params["shear"] = (x_shear, y_shear)
        else:
            x_shear = 0.0
            y_shear = 0.0

        # PERSPECTIVE
        if apply_with_probability(params["perspective_p"]) and (
            params.get("perspective_std", 0) > 0
            or params.get("perspective_delta", 0) > 0
            or params.get("perspective_x_delta", 0) > 0
            or params.get("perspective_y_delta", 0) > 0
        ):
            # image center projective deformation
            if "perspective_std" in params:
                x_perspective = (
                    np.random.normal(0, params["perspective_std"]) / patch_shape[1]
                )
                y_perspective = (
                    np.random.normal(0, params["perspective_std"]) / patch_shape[0]
                )
            else:
                assert (
                    "perspective_delta" in params
                    and "perspective_x_delta" in params
                    and "perspective_y_delta" in params
                    and "perspective_anisotropic_ratio" in params
                )
                if np.random.random() < params["perspective_anisotropic_ratio"]:
                    x_perspective = (
                        (np.random.random() * 2 - 1)
                        * params["perspective_x_delta"]
                        / patch_shape[1]
                    )
                    y_perspective = (
                        (np.random.random() * 2 - 1)
                        * params["perspective_y_delta"]
                        / patch_shape[0]
                    )
                else:
                    x_perspective = (
                        (np.random.random() * 2 - 1)
                        * params["perspective_delta"]
                        / patch_shape[1]
                    )
                    y_perspective = (
                        (np.random.random() * 2 - 1)
                        * params["perspective_delta"]
                        / patch_shape[0]
                    )
            H_params["perspective"] = (x_perspective, y_perspective)
        else:
            x_perspective = 0.0
            y_perspective = 0.0

        # ROTATION
        if apply_with_probability(params["angle_p"]) and (
            params.get("angle_std", 0) > 0 or params.get("angle_delta", 0) > 0
        ):
            # image center rotation
            if "angle_std" in params:
                alpha_degrees = np.random.normal(0, params["angle_std"])
            else:
                alpha_degrees = (np.random.random() * 2 - 1) * params["angle_delta"]
            H_params["rotation"] = alpha_degrees
        else:
            alpha_degrees = 0.0

        # SCALE
        if apply_with_probability(params["scale_p"]) and (
            params.get("scale_std", 0) > 0
            or params.get("scale_delta", 0) > 0
            or params.get("scale_x_delta", 0) > 0
            or params.get("scale_y_delta", 0) > 0
        ):
            if "scale_std" in params:
                x_scale = 1 + np.random.normal(0, params["scale_std"])
                y_scale = 1 + np.random.normal(0, params["scale_std"])
            else:
                # get only smaller images (to be sure that the patch fits the image)
                assert (
                    "scale_delta" in params
                    and "scale_x_delta" in params
                    and "scale_y_delta" in params
                    and "scale_anisotropic_ratio" in params
                )
                if np.random.random() < params["scale_anisotropic_ratio"]:
                    x_scale = 1 - (np.random.random() * params["scale_x_delta"])
                    y_scale = 1 - (np.random.random() * params["scale_y_delta"])
                else:
                    x_scale = 1 - (np.random.random() * params["scale_delta"])
                    y_scale = 1 - (np.random.random() * params["scale_delta"])
                # # get both bigger or smaller images
                # x_scale = 1 + (np.random.random() * 2 - 1) * params['scale_delta']
                # y_scale = 1 + (np.random.random() * 2 - 1) * params['scale_delta'] if params.get('scale_anisotropic') else x_scale
            H_params["scale"] = (x_scale, y_scale)
        else:
            x_scale = 1.0
            y_scale = 1.0

        # TRANSLATION
        if apply_with_probability(params["translation_p"]) and (
            params.get("translation_std", 0) > 0
            or params.get("translation_delta", 0) > 0
        ):
            if "translation_std" in params:
                x_translation = (
                    np.random.normal(0, params["translation_std"]) * patch_shape[1]
                )
                y_translation = (
                    np.random.normal(0, params["translation_std"]) * patch_shape[0]
                )
            else:
                x_translation = (
                    (np.random.random() * 2 - 1)
                    * params["translation_delta"]
                    * patch_shape[1]
                )
                y_translation = (
                    (np.random.random() * 2 - 1)
                    * params["translation_delta"]
                    * patch_shape[0]
                )
            H_params["translation"] = (x_translation, y_translation)
        else:
            x_translation = 0.0
            y_translation = 0.0

        H = generate_homography_for_patch_augmentation(
            patch_shape,
            patch_center,
            alpha_degrees=alpha_degrees,
            x_scale=x_scale,
            y_scale=y_scale,
            x_shear=x_shear,
            y_shear=y_shear,
            x_perspective=x_perspective,
            y_perspective=y_perspective,
            x_translation=x_translation,
            y_translation=y_translation,
        )

        # backproject the four patch corners
        points = np.array(
            [
                [0, 0],
                [patch_shape[1], 0],
                [patch_shape[1], patch_shape[0]],
                [0, patch_shape[0]],
            ]
        )
        points_backprojected = warp_points_numpy(points, H)
        # check if the polygon is convex
        if not is_convex(points_backprojected):
            print("Warning: generated homography is not convex")
            continue

        if params["allow_results_with_padding"]:
            break
        else:
            # check if all the points are inside the image
            if points_in_image(points_backprojected, source_img_shape):
                break
            else:
                print(
                    "Warning: one of the corners of the augmented patch is outside the source image. "
                    "Increase the source image size and margins, "
                    "decrease the patch size or the augmentation parameters."
                )
                continue

    # noinspection PyUnboundLocalVariable
    return H, H_params


def my_warp_perspective(
    img: Tensor,
    hom: Tensor,
    shape: Union[Tuple[int, int], Tensor],
    mode: str = "bilinear",
) -> Tensor:
    """the provided hom must map from the input to the output"""
    B, C, H, W = img.shape
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, shape[1]), torch.arange(0, shape[0]), indexing="xy"
            ),
            dim=-1,
        )[None]
        + 0.5
    )  # 1,H_out,W_out,2
    grid = grid.to(hom.dtype)
    grid_warped = warp_points(grid.view(1, -1, 2), torch.inverse(hom)).view(
        B, shape[0], shape[1], 2
    )  # 1,H,W,2
    grid_warped_normalized = grid_warped / (0.5 * torch.tensor([W, H]))[None] - 1
    output = F.grid_sample(
        img.to(hom.dtype),
        grid_warped_normalized,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    return output.to(img.dtype)


def rotate_image_and_crop_without_black_borders(
    img: Tensor, angle_degrees: float, mode: str = "bilinear"
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        img:
            C,H,W
        angle_degrees: the rotation angle in degrees
        mode: the interpolation mode
    Returns:
        img_rotated:
            C,H_rotated,W_rotated
        hom_rotation:
            3,3
        left_top: the coordinate of the top-left corner given in (x,y) order
            2
    """
    assert img.ndim == 3, f"img must be C,H,W, but got {img.shape}"

    matrix_rotation = rot_mat(angle_degrees / 180.0 * np.pi, dtype=torch.double)
    matrix_translation = transl_mat(
        (img.shape[-1] / 2, img.shape[-2] / 2), dtype=torch.double
    )
    hom_rotation = matrix_translation @ matrix_rotation @ matrix_translation.inverse()

    # find the largest possible rectangle that fits in the rotated image such that there are no black borders
    W_output, H_output, W_border, H_border = rotatedRectWithMaxArea(
        img.shape[-1], img.shape[-2], angle_degrees / 180.0 * np.pi
    )
    W_output, H_output, W_border, H_border = (
        int(W_output),
        int(H_output),
        int(W_border),
        int(H_border),
    )
    # print(f'angle: {angle_degrees},  {H_output},{W_output},  borders: {H_border},{W_border}')

    # correct the homography with this translation
    matrix_border = transl_mat((-W_border, -H_border), dtype=torch.double)
    hom_rotation_without_border = matrix_border @ hom_rotation
    # # NOT APPLY ANY CROPPING
    # hom_rotation_without_border = hom_rotation
    # H_output, W_output = img.shape[-2], img.shape[-1]

    # apply the rotation to the image
    img_rotated = my_warp_perspective(
        img[None],
        hom_rotation_without_border[None],
        shape=(H_output, W_output),
        mode=mode,
    )[0]

    return img_rotated, hom_rotation_without_border, torch.tensor([W_border, H_border])
