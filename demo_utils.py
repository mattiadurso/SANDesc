"""Helper functions for the demo notebook."""

import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

TensorOrArray = torch.Tensor | np.ndarray


def unproject_points2d(
    points: TensorOrArray, K: TensorOrArray, remove_last: bool = True
) -> Tensor:
    """Unproject 2D points to 3D points."""
    points = to_torch(points, b=False)
    K = to_torch(K, b=False)

    points = to_homogeneous(points)
    points_unprojected = (K.inverse() @ points.permute(-1, -2)).permute(-1, -2)

    if remove_last:
        points_unprojected = points_unprojected[:, :2] / points_unprojected[:, 2:]
        return points_unprojected.reshape(-1, 2)

    points_unprojected = points_unprojected / points_unprojected[:, 2:]
    return points_unprojected.reshape(-1, 3)


def to_homogeneous(vector: TensorOrArray) -> Tensor:
    """Convert a 2D vector to homogeneous coordinates."""
    vector = to_torch(vector, b=False)
    if vector.shape[1] == 2:
        vector = torch.hstack([vector, torch.ones_like(vector)[..., :1]])
    return vector.float()


def compute_epipolar_lines_coeff(
    E: TensorOrArray,  # bx3x3
    points: TensorOrArray,  # bxNx2
    K: TensorOrArray | None = None,  # bx3x3
) -> Tensor:
    """Compute epipolar line coefficients from a matrix and points.

    Computes them from the essential/fundamental matrix and the points. It is
    needed to unproject points if using the Essential matrix.

    Args:
        E: essential matrix
        points: points in the image
        K: intrinsics matrix. If not provide E is assumed to be the
           fundamental matrix.

    Returns:
        epi_lines: epipolar lines coefficients

    """
    points = to_torch(points, b=False)
    E = to_torch(E, b=False)[0]
    if K is not None:
        K = to_torch(K, b=False)

    if K is not None:
        points = unproject_points2d(points, K, remove_last=False)
    else:
        points = to_homogeneous(points)

    return (E @ points.T).T  # epipolar coefficients [a,b,c] for each point


def distance_line_points_parallel(line: Tensor, points: Tensor) -> Tensor:
    """Compute point-to-line distances.

    Args:
        line: tensor [1,3], [3], [3,1].
        points: tensor [N,2].
    """
    a, b, c = line.flatten()
    x, y = points[:, 0], points[:, 1]
    return torch.abs(a * x + b * y + c) / (a**2 + b**2) ** 0.5


def is_torch(vector: object) -> bool:
    """Check if a vector is a torch tensor."""
    return bool(isinstance(vector, torch.Tensor))


def to_torch(vector_: TensorOrArray, b: bool = True) -> Tensor:
    """Convert a numpy array to a torch tensor. Eventually add batch size."""
    vector = deepcopy(vector_)
    if not is_torch(vector):
        vector = torch.tensor(vector)

    if b and len(vector.shape) < 3:
        vector = vector.unsqueeze(0)

    return vector.float()


def compute_fundamental_from_relative_motion(
    R: TensorOrArray, t: TensorOrArray, K0: TensorOrArray, K1: TensorOrArray
) -> Tensor:
    """Compute the fundamental matrix from the relative rotation and translation.

    Args:
        R: relative rotation matrix
        t: relative translation vector
        K0: intrinsics matrix of image 0
        K1: intrinsics matrix of image 1
    Returns:
        Fm: fundamental matrix

    """
    R, t = to_torch(R), to_torch(t, b=False)
    K0, K1 = to_torch(K0, b=True), to_torch(K1, b=True)
    Em = compute_essential_from_relative_motion(R, t)
    return torch.bmm(K1.permute(0, 2, 1).inverse(), torch.bmm(Em, K0.inverse()))


def compute_essential_from_relative_motion(
    R: TensorOrArray, t: TensorOrArray
) -> Tensor:
    """Compute the essential matrix from the relative rotation and translation.

    Args:
        R: relative rotation matrix
        t: relative translation vector
    Returns:
        Em: essential matrix

    """
    R = to_torch(R)
    t = to_torch(t, b=False)

    Tx = vector_to_skew_symmetric_matrix(t)
    return Tx @ R


def vector_to_skew_symmetric_matrix(t: Tensor) -> Tensor:
    """Convert a 3D vector to its skew-symmetric matrix.

    Args:
        t: torch.Tensor of shape (3,) or (1, 3)

    Returns:
        Tx: torch.Tensor of shape (3, 3)

    """
    t = t.flatten()
    Tx = torch.zeros(3, 3, dtype=t.dtype, device=t.device)
    Tx[0, 1] = -t[2]
    Tx[0, 2] = t[1]
    Tx[1, 0] = t[2]
    Tx[1, 2] = -t[0]
    Tx[2, 0] = -t[1]
    Tx[2, 1] = t[0]
    return Tx


def plot_imgs(
    images: list[TensorOrArray],
    titles: list[str] | None = None,
    rows: int = 1,
) -> None:
    """Plot images in a grid with the specified number of rows.

    Args:
        images (list of torch.Tensor or numpy.ndarray): List of images to plot.
        titles (list of str, optional): List of titles for each image.
        rows (int, optional): Number of rows in the grid. Default is 1.

    """
    # Calculate number of columns based on the number of rows
    cols = -(-len(images) // rows)  # Ceiling division to handle uneven grids

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 5 * rows))

    # Flatten axes for easy iteration, in case rows or cols == 1
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx]

            # Check if image is in C, H, W format and convert to H, W, C if needed
            is_chw = (
                isinstance(img, torch.Tensor)
                and img.ndim == 3
                and img.shape[0] in [1, 3]
            )
            if is_chw:  # C, H, W
                img = img.permute(1, 2, 0).cpu().numpy()  # Channels first to last

            # Determine colormap based on number of channels
            if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
                cmap = "gray"
            else:
                cmap = None

            ax.imshow(img.squeeze(), cmap=cmap)
            ax.axis("off")

            # Set title if provided
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])
        else:
            ax.axis("off")  # Hide unused axes

    plt.tight_layout()
    plt.show()


def plot_imgs_and_kpts(
    img1: TensorOrArray,
    img2: TensorOrArray,
    kpt1: np.ndarray,
    kpt2: np.ndarray,
    space: int = 100,
    matches: bool = True,
    index: bool = False,
    sample_points: int = 32,
    pad: bool = False,
    figsize: tuple[int, int] = (10, 5),
    axis: bool = True,
    scatter: bool = True,
    highlight_bad_matches: bool | None = None,
    F_gt: TensorOrArray | None = None,
    plot_name: str | None = None,
    reth: int = 5,
    text: str | None = None,
    text_font: int = 12,
    matches_linewidth: int = 1,
) -> None:
    """Plot two images side by side with keypoints and optional matches."""
    assert img1.shape[-1] == img2.shape[-1], "Images must have the same channels"
    assert img1.shape[-1] in [1, 3], "Images must be RGB"
    c = img1.shape[-1]
    # check if images are numpy, then to tensor
    if isinstance(img1, np.ndarray):
        img1 = torch.tensor(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.tensor(img2)

    def pad_to_height(
        img: Tensor, target_h: int, pad_color: tuple[int, int, int]
    ) -> tuple[Tensor, int]:
        h, w, c = img.shape
        if h >= target_h:
            return img, 0
        total_pad = target_h - h
        pad_top = total_pad // 2

        # build pad canvas with desired color
        color_tensor = torch.tensor(pad_color, dtype=img.dtype, device=img.device).view(
            1, 1, 3
        )
        padded = color_tensor.expand(target_h, w, 3).clone()
        padded[pad_top : pad_top + h] = img
        return padded, pad_top

    # Determine target height and pad both images
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    target_h = max(h1, h2)

    img1, offset1 = pad_to_height(img1, target_h, (255, 255, 255))
    img2, offset2 = pad_to_height(img2, target_h, (255, 255, 255))

    # Adjust keypoints for vertical padding (y coordinate)
    kpt1 = deepcopy(kpt1).astype(np.float32)
    kpt2 = deepcopy(kpt2).astype(np.float32)
    kpt1[:, 1] += offset1
    kpt2[:, 1] += offset2

    if pad:
        # find smaller image and put it in a white canvas of the size of the
        # bigger image
        if img1.shape[0] < img2.shape[0]:
            img1 = torch.cat(
                (
                    img1,
                    torch.ones((img2.shape[0] - img1.shape[0], img1.shape[1], c)).int()
                    * 255,
                ),
                dim=0,
            )
        else:
            img2 = torch.cat(
                (
                    img2,
                    torch.ones((img1.shape[0] - img2.shape[0], img2.shape[1], c)).int()
                    * 255,
                ),
                dim=0,
            )

        if img1.shape[1] < img2.shape[1]:
            img1 = torch.cat(
                (
                    img1,
                    torch.ones((img1.shape[0], img2.shape[1] - img1.shape[1], c)).int()
                    * 255,
                ),
                dim=1,
            )
        else:
            img2 = torch.cat(
                (
                    img2,
                    torch.ones((img2.shape[0], img1.shape[1] - img2.shape[1], c)).int()
                    * 255,
                ),
                dim=1,
            )

    white = torch.ones((img1.shape[0], space, c)).int() * 255
    concat = torch.cat((img1, white, img2), dim=1).int()

    plt.figure(figsize=figsize)
    plt.imshow(concat)

    if scatter:
        if sample_points and sample_points < len(kpt1):
            kpt1 = kpt1[:: len(kpt1) // sample_points]
            kpt2 = kpt2[:: len(kpt2) // sample_points]

        if index:
            for i, (x, y) in enumerate(kpt1):
                plt.text(x, y, c="w", s=str(i), fontsize=6, ha="center", va="center")
            for i, (x, y) in enumerate(kpt2):
                plt.text(
                    x + img1.shape[1] + space,
                    y,
                    c="w",
                    s=str(i),
                    fontsize=6,
                    ha="center",
                    va="center",
                )

        plt.scatter(kpt1[:, 0], kpt1[:, 1], c="r", s=2)
        plt.scatter(kpt2[:, 0] + img1.shape[1] + space, kpt2[:, 1], c="r", s=2)

    if matches:
        if highlight_bad_matches is not None:
            points1 = to_torch(kpt1, b=False)
            points2 = to_torch(kpt2, b=False)
            E12 = to_torch(F_gt)
            E21 = E12.permute(0, 2, 1)

            epilines_A = compute_epipolar_lines_coeff(E12, points1)
            epilines_B = compute_epipolar_lines_coeff(E21, points2)

            repr_err_A = [
                distance_line_points_parallel(epilines_A[i], points2[i][None]).item()
                for i in range(epilines_A.shape[0])
            ]
            repr_err_B = [
                distance_line_points_parallel(epilines_B[i], points1[i][None]).item()
                for i in range(epilines_B.shape[0])
            ]

            reth = 5
            good_matches = (torch.tensor(repr_err_A) <= reth) & (
                torch.tensor(repr_err_B) <= reth
            )

            kpts1_matched_good = kpt1[good_matches]
            kpts2_matched_good = kpt2[good_matches]

            kpts1_matched_bad = kpt1[~good_matches]
            kpts2_matched_bad = kpt2[~good_matches]
            print(f"Reprojection error threshold: {reth} pixels")
            print(f"Inliers: {kpts1_matched_good.shape[0]}/{kpt1.shape[0]}")

            for i in range(kpts1_matched_good.shape[0]):
                plt.plot(
                    [
                        kpts1_matched_good[i, 0],
                        kpts2_matched_good[i, 0] + img1.shape[1] + space,
                    ],
                    [kpts1_matched_good[i, 1], kpts2_matched_good[i, 1]],
                    c="g",
                    linewidth=matches_linewidth,
                    alpha=0.85,
                )

            for i in range(kpts1_matched_bad.shape[0]):
                plt.plot(
                    [
                        kpts1_matched_bad[i, 0],
                        kpts2_matched_bad[i, 0] + img1.shape[1] + space,
                    ],
                    [kpts1_matched_bad[i, 1], kpts2_matched_bad[i, 1]],
                    c="r",
                    linewidth=matches_linewidth,
                    alpha=0.85,
                )

        else:
            for i in range(kpt1.shape[0]):
                plt.plot(
                    [kpt1[i, 0], kpt2[i, 0] + img1.shape[1] + space],
                    [kpt1[i, 1], kpt2[i, 1]],
                    c="b",
                    linewidth=0.25,
                    alpha=0.75,
                )

    if text is not None:
        # plot text just above the image, align text to the top left corner
        plt.text(
            0.5,
            1.01,
            text,
            transform=plt.gca().transAxes,
            fontsize=text_font,
            ha="center",
            va="bottom",
        )
    plt.axis("off" if axis else "on")
    # save
    plt.tight_layout()
    if plot_name is not None:
        # timestamp
        time_ = time.strftime("%Y-%m-%d_%H-%M-%S")
        plot_name = plot_name if plot_name is not None else f"plot_{time_}"
        plt.savefig(plot_name + ".png", bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()


def compute_relative_pose(
    R1: Tensor, t1: Tensor, R2: Tensor, t2: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute the relative rotation and translation between two poses."""
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans


def load_image(path: str, scaling: float = 1.0) -> Tensor:
    """Load an image as a float32 tensor in [0, 1].

    Resizes if needed and crops to a multiple of 16.
    """
    img = read_image_to_torch(str(path)) / 255.0  # 3, H, W, float32 [0, 1]
    # resize if needed
    if scaling != 1.0:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            scale_factor=1 / scaling,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    # crop to multiple of 16
    return crop_to_multiple_of(img, multiple_of=16)


def read_image_to_torch(path: str) -> Tensor:
    """Read image with OpenCV and convert to RGB.

    Returns a uint8 tensor CxHxW in [0,255].
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # HxWxC (BGR) or HxW (gray)
    if img is None:
        raise FileNotFoundError(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = torch.from_numpy(img)  # HxWxC, uint8/uint16
    return t.permute(2, 0, 1).contiguous()  # CxHxW


def crop_to_multiple_of(img: TensorOrArray, multiple_of: int = 16) -> TensorOrArray:
    """Crop an image (array or tensor) so H and W are multiples of a number."""
    if isinstance(img, np.ndarray):
        H, W = img.shape[:2]
        new_H = (H // multiple_of) * multiple_of
        new_W = (W // multiple_of) * multiple_of
        return img[:new_H, :new_W, :]

    if isinstance(img, torch.Tensor):
        H, W = img.shape[-2:]
        new_H = (H // multiple_of) * multiple_of
        new_W = (W // multiple_of) * multiple_of
        return img[..., :new_H, :new_W]
    raise TypeError("Unsupported image type")


def estimate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    norm_thresh: float,
    conf: float = 0.999999,
    maxIters: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Estimate pose using essential matrix."""
    # Quick safeguard in case K0/K1 still have a batch dimension of (1, 3, 3)
    K0 = K0.squeeze()
    K1 = K1.squeeze()

    if len(kpts0) < 5:
        return None

    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])

    kpts0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    kpts1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T

    E, mask = cv2.findEssentialMat(
        kpts0,
        kpts1,
        cameraMatrix=np.eye(3),
        threshold=norm_thresh,
        prob=conf,
        maxIters=maxIters,
        method=cv2.RANSAC,
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) // 3):
            # THE FIX: Create a fresh copy of the RANSAC mask for this iteration
            curr_mask = mask.copy()

            n, R, t, mask_out = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=curr_mask
            )

            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask_out.ravel() > 0)

    return ret


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute angle error between two rotation matrices."""
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle error between two vectors."""
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(
    T_0to1: np.ndarray, R: np.ndarray, t: np.ndarray
) -> tuple[float, float]:
    """Compute pose error given ground truth and estimated pose."""
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R
