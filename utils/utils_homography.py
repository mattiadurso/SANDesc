"""homography utils."""

import logging
import math
from collections.abc import Collection

import numpy as np
import torch
import torch.nn.functional as F
from numpy import array
from torch import Tensor

from utils.utils_2D import (
    compute_quadrilateral_area_from_corners,
    filter_outside,
    mutual_nearest_neighbors_from_dist_matrices,
    rotatedRectWithMaxArea,
)

log = logging.getLogger(__name__)


def apply_with_probability(probability: float) -> bool:
    """Return True or False with the specified probability.

    Args:
        probability: the specified probability.

    Returns:
        True or False with the specified probability.
    """
    return np.random.rand() < probability


def rot_mat(
    alpha: float, dtype: torch.dtype = None, device: str | torch.device = None
) -> Tensor:
    """Return a 3x3 rotation matrix for angle ``alpha`` (radians)."""
    return torch.tensor(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def transl_mat(
    xy: Tensor | tuple,
    dtype: torch.dtype = None,
    device: str | torch.device = None,
) -> Tensor:
    """Return a 3x3 translation matrix for offset ``xy``."""
    return torch.tensor(
        [[1.0, 0.0, xy[0]], [0.0, 1.0, xy[1]], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )


def transl_mat_numpy(xy: np.ndarray | tuple) -> np.ndarray:
    """Return a 3x3 translation matrix for offset ``xy`` (numpy)."""
    return np.array([[1.0, 0.0, xy[0]], [0.0, 1.0, xy[1]], [0.0, 0.0, 1.0]])


def rot_mat_numpy(alpha: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for angle ``alpha`` (numpy)."""
    return np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def scale_mat(scale: float | Tensor | tuple) -> Tensor:
    """Return a 3x3 scale matrix (isotropic or per-axis)."""
    s = scale if isinstance(scale, Collection) else (scale, scale)
    return torch.tensor([[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 1.0]])


def scale_mat_numpy(scale: float | np.ndarray | tuple) -> np.ndarray:
    """Return a 3x3 scale matrix (isotropic or per-axis, numpy)."""
    s = scale if isinstance(scale, Collection) else (scale, scale)
    return np.array([[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, 1.0]])


def shear_mat_numpy(shear: np.ndarray | tuple) -> np.ndarray:
    """Return a 3x3 shear matrix (numpy)."""
    return np.array([[1.0, shear[0], 0.0], [shear[1], 1.0, 0.0], [0.0, 0.0, 1.0]])


def perspective_mat_numpy(perspective: np.ndarray | tuple) -> np.ndarray:
    """Return a 3x3 perspective matrix (numpy)."""
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective[0], perspective[1], 1.0]]
    )


def warp_points(
    xy: Tensor,
    H: Tensor,
    img_shape: np.ndarray | Tensor | tuple[int, int] = None,
    border: int = 0,
) -> Tensor:
    """Warp the points using the provided homography matrix.

    Args:
        xy: input points coordinates with order (x, y)
           B,n,2
        H: input homography
           B,3,3
        img_shape: if provided, the points that project out of shape are set
            to 'nan'.
        border: together with img_shape, set to nan the points that are closer
            to the border than this.

    Returns:
        The projected points
           B,n,2.
    """
    assert xy.shape[0] == H.shape[0]
    assert xy.shape[2] == 2
    assert H.shape[1] == 3 and H.shape[2] == 3
    if isinstance(img_shape, (Tensor, np.ndarray)):
        assert img_shape.shape == (2,)

    xy_hom = torch.cat(
        (
            xy,
            torch.ones((xy.shape[0], xy.shape[1], 1), dtype=xy.dtype, device=xy.device),
        ),
        dim=2,
    ).to(H.dtype)  # B,n,3
    xy_proj_hom = xy_hom @ H.to(xy.device).permute(0, 2, 1)  # B,n,3
    xy_proj = xy_proj_hom[:, :, 0:2] / xy_proj_hom[:, :, 2:3]  # B,n,2

    if img_shape is not None:
        xy_proj = filter_outside(xy_proj, img_shape, border)

    return xy_proj.to(xy.dtype)


def warp_points_numpy(points: array, H: array) -> array:
    """Warp the points using the provided homography matrix (numpy).

    Args:
        points: the input coordinates
            nx2
        H: input homography
            3x3

    Returns:
        The projected points
            nx2.
    """
    points_dst = H.dot(np.concatenate((points.T, np.ones((1, points.shape[0])))))
    points_dst = points_dst[0:2, :] / points_dst[2, :]
    return points_dst.T


def get_dist_matrix(
    xy0: Tensor,
    xy1: Tensor,
    H0_1: Tensor,
    img1_shape: tuple[int, int] | None = None,
    border: int = 0,
    return_projected: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute the distance matrix between projected img0 points and img1.

    Distances are between the keypoints projected from img0 (rows) to img1
    (columns) using the provided homography.

    Args:
        xy0: input points in img0
            Bxn0x2
        xy1: input points in img1
            Bxn1x2
        H0_1: homography that warps img0 in img1
            Bx3x3
        img1_shape: the shape of img1.
        border: together with img1_shape, used to invalidate the points that
            are close to the image borders.
        return_projected: additionally return the projected points.

    Returns:
        Distance computed in img1 between the projected points xy0_proj and the
        points in img1. CAN HAVE 'nan' if img_shape is provided and the points
        project out of the second image
            Bxn0xn1.
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
    img0_shape: tuple[int, int] | None = None,
    img1_shape: tuple[int, int] | None = None,
    border: int = 0,
    return_distances_and_projected: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | Tensor:
    """Compute the GT matching matrix using the homography img0 -> img1.

    Args:
        xy0: input points in img0
            Bxn0x2
        xy1: input points in img1
            Bxn1x2
        H0_1: homography that warps img0 in img1
            Bx3x3
        thr: threshold for considering a match valid.
        img0_shape: the shape of img0; points that project out are invalid.
        img1_shape: the shape of img1; points that project out are invalid.
        border: keypoints closer to the border than this value will not
            generate valid matches.
        return_distances_and_projected: additionally return computed distances.

    Returns:
        GT_matching_matrix: the resulting GT matching matrix
            B x n0+1 x n1+1   torch.bool.
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


def points_in_image(xy: np.ndarray, img_shape: tuple[int, int]) -> bool:
    """Check if all the point coordinates are contained inside the image.

    Args:
        xy: input points coordinate with (x, y).
        img_shape: the image shape where all the points must be inside.

    Returns:
        bool: True if all the points lie inside the image.
    """
    return bool(
        (xy[:, 0] > 0).all()
        and (xy[:, 0] < img_shape[1]).all()
        and (xy[:, 1] > 0).all()
        and (xy[:, 1] < img_shape[0]).all()
    )


def is_convex(xy: np.ndarray) -> bool:
    """Check if the polygon is convex."""
    N = xy.shape[0]

    # direction of cross product of previous edges
    direction_old = 0

    for i in range(N):
        # compute two consecutive edges
        vector = np.array([xy[(i + 1) % N], xy[(i + 2) % N]]) - xy[i]

        # 2-D scalar cross product (z-component); np.cross on 2-D vectors is
        # deprecated in NumPy 2.0.
        direction_new = vector[0][0] * vector[1][1] - vector[0][1] * vector[1][0]
        # if direction of cross product of all adjacent edges are not same
        if direction_new * direction_old < 0:
            return False
        # update old
        direction_old = direction_new
    return True


def generate_homography_for_patch_augmentation(
    patch_shape: np.ndarray | Tensor,
    patch_center: np.ndarray | Tensor,
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
    """Compose a homography that augments a patch around its center.

    Args:
        patch_shape: the (height, width) of the patch.
        patch_center: the (x, y) center the transforms are applied around.
        alpha_degrees: rotation angle in degrees.
        x_shear: horizontal shear factor.
        y_shear: vertical shear factor.
        x_translation: horizontal translation.
        y_translation: vertical translation.
        x_scale: horizontal scale factor.
        y_scale: vertical scale factor.
        y_perspective: vertical perspective factor.
        x_perspective: horizontal perspective factor.

    Returns:
        The composed 3x3 homography matrix.
    """
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

    # translate back and forward so all the transformations are applied at the
    # center of the patch
    t = np.array([[1, 0, patch_center[0]], [0, 1, patch_center[1]], [0, 0, 1]])
    H = t @ H @ np.linalg.inv(t)

    # correct the homography so that when warping the image, the patch_center
    # coordinate ends exactly in the center of the patch
    translation = transl_mat_numpy(
        (patch_center[0] - patch_shape[1] // 2, patch_center[1] - patch_shape[0] // 2)
    )
    return H @ translation


def _sample_shear(params: dict) -> tuple[float, float, bool]:
    """Sample (x, y) shear. Returns (x_shear, y_shear, applied)."""
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
        return x_shear, y_shear, True
    return 0.0, 0.0, False


def _sample_perspective(
    params: dict, patch_shape: tuple[int, int] | np.ndarray
) -> tuple[float, float, bool]:
    """Sample (x, y) perspective. Returns (x_persp, y_persp, applied)."""
    if apply_with_probability(params["perspective_p"]) and (
        params.get("perspective_std", 0) > 0
        or params.get("perspective_delta", 0) > 0
        or params.get("perspective_x_delta", 0) > 0
        or params.get("perspective_y_delta", 0) > 0
    ):
        if "perspective_std" in params:
            std = params["perspective_std"]
            x_perspective = np.random.normal(0, std) / patch_shape[1]
            y_perspective = np.random.normal(0, std) / patch_shape[0]
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
        return x_perspective, y_perspective, True
    return 0.0, 0.0, False


def _sample_rotation(params: dict) -> tuple[float, bool]:
    """Sample a rotation angle in degrees. Returns (alpha_degrees, applied)."""
    if apply_with_probability(params["angle_p"]) and (
        params.get("angle_std", 0) > 0 or params.get("angle_delta", 0) > 0
    ):
        if "angle_std" in params:
            alpha_degrees = np.random.normal(0, params["angle_std"])
        else:
            alpha_degrees = (np.random.random() * 2 - 1) * params["angle_delta"]
        return alpha_degrees, True
    return 0.0, False


def _sample_scale(params: dict) -> tuple[float, float, bool]:
    """Sample (x, y) scale. Returns (x_scale, y_scale, applied)."""
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
        return x_scale, y_scale, True
    return 1.0, 1.0, False


def _sample_translation(
    params: dict, patch_shape: tuple[int, int] | np.ndarray
) -> tuple[float, float, bool]:
    """Sample (x, y) translation. Returns (x_transl, y_transl, applied)."""
    if apply_with_probability(params["translation_p"]) and (
        params.get("translation_std", 0) > 0 or params.get("translation_delta", 0) > 0
    ):
        if "translation_std" in params:
            std = params["translation_std"]
            x_translation = np.random.normal(0, std) * patch_shape[1]
            y_translation = np.random.normal(0, std) * patch_shape[0]
        else:
            delta = params["translation_delta"]
            x_translation = (np.random.random() * 2 - 1) * delta * patch_shape[1]
            y_translation = (np.random.random() * 2 - 1) * delta * patch_shape[0]
        return x_translation, y_translation, True
    return 0.0, 0.0, False


def _translation_only_homography(
    patch_center: np.ndarray, patch_shape: tuple[int, int] | np.ndarray
) -> tuple[np.ndarray, dict]:
    """Fallback homography with only the centering translation."""
    translation = (
        patch_center[0] - patch_shape[1] // 2,
        patch_center[1] - patch_shape[0] // 2,
    )
    return transl_mat_numpy(translation), {"translation": translation}


def _patch_corners_valid(
    homography: np.ndarray,
    patch_shape: tuple[int, int] | np.ndarray,
    source_img_shape: tuple[int, int] | np.ndarray,
    allow_padding: bool,
) -> bool:
    """Check the backprojected patch corners are convex and (optionally) inside."""
    points = np.array(
        [
            [0, 0],
            [patch_shape[1], 0],
            [patch_shape[1], patch_shape[0]],
            [0, patch_shape[0]],
        ]
    )
    points_backprojected = warp_points_numpy(points, homography)
    if not is_convex(points_backprojected):
        log.warning("generated homography is not convex")
        return False
    if allow_padding:
        return True
    if points_in_image(points_backprojected, source_img_shape):
        return True
    log.warning(
        "one of the corners of the augmented patch is outside the source "
        "image. Increase the source image size and margins, decrease the "
        "patch size or the augmentation parameters."
    )
    return False


def sample_homography(
    patch_center: np.ndarray,
    patch_shape: tuple[int, int] | np.ndarray,
    source_img_shape: tuple[int, int] | np.ndarray,
    params: dict[str, float],
    max_n_iterations: int = 1000,
) -> tuple[np.ndarray, dict]:
    """Sample a random homography using the provided parameters.

    Args:
        patch_center: the coordinates of the patch center given as (x, y).
        patch_shape: the patch shape that must completely fall inside the warped
            image given as (H, W).
        source_img_shape: the input image shape as (H, W).
        params: a dict that must contain
            'angle_delta': rotation standard deviation in degrees,
            'angle_std': rotation standard deviation in degrees,
            'angle_p': probability of applying rotation,
            'translation_std': translation standard deviation given in relation
                to the patch_shape (1.0 means 100% of the patch dimension),
            'translation_delta': translation max and min value for uniform
                sampling given in relation to the patch_shape (1.0 means 100%
                of the patch dimension),
            'translation_p': probability of applying translation,
            'shear_std': shear standard deviation,
            'shear_delta': shear max and min value for uniform sampling
                (reasonable values are up to 2.0),
            'shear_p': probability of applying shear
                (reasonable values are up to 2.0),
            'scale_std': scale standard deviation,
            'scale_delta': scale max and min value for uniform sampling,
            'scale_p': probability of applying scale,
            'scale_anisotropic': if true scale for x and y can be different,
            'perspective_std': perspective standard deviation. The code corrects
                for scale and translation (reasonable values are up to 0.5, max
                value is abs(perspective_x) + abs(perspective_y) = 2.0),
            'perspective_delta': perspective min and max for uniform sampling.
                The code corrects for scale and translation (reasonable values
                are up to 0.5, max value is abs(perspective_x) +
                abs(perspective_y) = 2.0),
            'perspective_p': probability of applying perspective,
            'allow_results_with_padding': if true enable homographies that would
                include areas outside of the image in the output patch.
        max_n_iterations: the number of iterations before exiting, returning the
            homography with just the translation.

    Returns:
        H: sampled 3x3 homography that maps from OUTPUT to INPUT.
        H_params: dict of selected parameters for the homography.
    """
    assert ("angle_std" in params) != ("angle_delta" in params), (
        "only std or delta must be provided"
    )
    assert ("translation_std" in params) != ("translation_delta" in params), (
        "only std or delta must be provided"
    )
    assert ("shear_std" in params) != ("shear_delta" in params), (
        "only std or delta must be provided"
    )
    assert ("scale_std" in params) != ("scale_delta" in params), (
        "only std or delta must be provided"
    )
    assert ("perspective_std" in params) != ("perspective_delta" in params), (
        "only std or delta must be provided"
    )

    H_params = {}

    for i in range(max_n_iterations):
        if i == max_n_iterations - 1:
            # reached the max number of iterations: return just the translation
            log.warning(
                "could not generate the homography, returning the homography "
                "with just the translation"
            )
            return _translation_only_homography(patch_center, patch_shape)

        x_shear, y_shear, shear_applied = _sample_shear(params)
        if shear_applied:
            H_params["shear"] = (x_shear, y_shear)

        x_perspective, y_perspective, persp_applied = _sample_perspective(
            params, patch_shape
        )
        if persp_applied:
            H_params["perspective"] = (x_perspective, y_perspective)

        alpha_degrees, rotation_applied = _sample_rotation(params)
        if rotation_applied:
            H_params["rotation"] = alpha_degrees

        x_scale, y_scale, scale_applied = _sample_scale(params)
        if scale_applied:
            H_params["scale"] = (x_scale, y_scale)

        x_translation, y_translation, translation_applied = _sample_translation(
            params, patch_shape
        )
        if translation_applied:
            H_params["translation"] = (x_translation, y_translation)

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

        if _patch_corners_valid(
            H, patch_shape, source_img_shape, params["allow_results_with_padding"]
        ):
            return H, H_params

    # unreachable: the loop either returns a valid H or the translation fallback
    return _translation_only_homography(patch_center, patch_shape)


def my_warp_perspective(
    img: Tensor,
    hom: Tensor,
    shape: tuple[int, int] | Tensor,
    mode: str = "bilinear",
) -> Tensor:
    """The provided hom must map from the input to the output."""
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
) -> tuple[Tensor, Tensor, Tensor]:
    """Rotate an image and crop it to the largest border-free rectangle.

    Args:
        img: input image with shape C,H,W.
        angle_degrees: the rotation angle in degrees.
        mode: the interpolation mode.

    Returns:
        img_rotated:
            C,H_rotated,W_rotated
        hom_rotation:
            3,3
        left_top: the coordinate of the top-left corner given in (x,y) order
            2.
    """
    assert img.ndim == 3, f"img must be C,H,W, but got {img.shape}"

    matrix_rotation = rot_mat(angle_degrees / 180.0 * np.pi, dtype=torch.double)
    matrix_translation = transl_mat(
        (img.shape[-1] / 2, img.shape[-2] / 2), dtype=torch.double
    )
    hom_rotation = matrix_translation @ matrix_rotation @ matrix_translation.inverse()

    # find the largest possible rectangle that fits in the rotated image
    # such that there are no black borders
    W_output, H_output, W_border, H_border = rotatedRectWithMaxArea(
        img.shape[-1], img.shape[-2], angle_degrees / 180.0 * np.pi
    )
    W_output, H_output, W_border, H_border = (
        int(W_output),
        int(H_output),
        int(W_border),
        int(H_border),
    )
    # print(f'angle: {angle_degrees}, {H_output},{W_output},'
    #       f' borders: {H_border},{W_border}')

    # correct the homography with this translation
    matrix_border = transl_mat((-W_border, -H_border), dtype=torch.double)
    hom_rotation_without_border = matrix_border @ hom_rotation

    # apply the rotation to the image
    img_rotated = my_warp_perspective(
        img[None],
        hom_rotation_without_border[None],
        shape=(H_output, W_output),
        mode=mode,
    )[0]

    return img_rotated, hom_rotation_without_border, torch.tensor([W_border, H_border])
