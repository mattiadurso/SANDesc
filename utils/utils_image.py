"""Image loading and preprocessing utilities."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from torch import Tensor


def crop_multiple_of(img: Tensor, multiple: int) -> Tensor:
    """Crop an image to dimensions that are multiples of a given number.

    The crop starts from the top-left corner such that both crop_H and crop_W
    are multiples of the provided number. They can differ, and are chosen as
    the closest to the original resolution.

    Args:
        img: input image tensor
            ...,H,W
        multiple: output dimension will be a multiple of this number.

    Returns:
        The output crop.
    """
    H, W = img.shape[-2:]

    H_crop = (H // multiple) * multiple
    W_crop = (W // multiple) * multiple

    return img[..., :H_crop, :W_crop]


def denormalize(
    img: Tensor,
    RGB_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    RGB_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tensor:
    """Denormalize image color given the parameters used to normalize it.

    Args:
        img: input image
            BxCxHxW or BxHxW or CxHxW or HxW
        RGB_mean: mean values used for the normalization.
        RGB_std: std values used for the normalization.

    Returns:
        The rescaled color image.
    """
    return torchvision.transforms.functional.normalize(
        img,
        mean=[
            -RGB_mean[0] / RGB_std[0],
            -RGB_mean[1] / RGB_std[1],
            -RGB_mean[2] / RGB_std[2],
        ],
        std=[1 / RGB_std[0], 1 / RGB_std[1], 1 / RGB_std[2]],
    )


def pad_and_cut_image(
    img: Tensor,
    shape: tuple[int, int],
    mode: str = "center",
    value: float = float("nan"),
    allow_cuts: bool = False,
) -> Tensor:
    """Pad the input image to the required shape, keeping it centered.

    Args:
        img: the input image
            [... x H x W]
        shape: the required shape as (H, W).
        value: the value used for the padding.
        mode: the padding mode, 'center' or 'bottom-right'.
        allow_cuts: whether to allow cutting the image or not.

    Returns:
        The padded image.
    """
    H, W = img.shape[-2:]
    assert mode in ["center", "bottom-right"]
    if allow_cuts:
        img = img.clone()[..., : min(shape[0], H), : min(shape[1], W)]
        H, W = img.shape[-2:]
    else:
        assert shape[0] >= H and shape[1] >= W

    pad_x = shape[1] - W
    pad_y = shape[0] - H
    if mode == "center":
        pad_l = pad_x // 2
        pad_r = pad_x - pad_l
        pad_t = pad_y // 2
        pad_b = pad_y - pad_t
    else:
        # bottom-right
        pad_l = 0
        pad_r = pad_x
        pad_t = 0
        pad_b = pad_y

    return F.pad(img, [pad_l, pad_r, pad_t, pad_b], value=value)


def gray_to_colormap(
    img: Tensor, cmap: str = "inferno", rescale: bool = True
) -> Tensor:
    """Convert a grayscale image to a specific matplotlib colormap.

    Args:
        img: the input batched grayscale images
            HxW or 1xHxW
        cmap: string that refers to a matplotlib colormap.
        rescale: whether to rescale the colors.

    Returns:
        The output RGB image
            3xHxW
    """
    assert img.ndim == 2 or img.ndim == 3
    if img.ndim == 3:
        assert img.shape[0] == 1
        img = img[0]

    cm = plt.get_cmap(cmap)
    img = img.cpu().detach().numpy()
    if rescale:
        img -= img.min()
        img /= img.max()

    color = cm(img)[:, :, :3]
    return torch.tensor(color, dtype=torch.float32).permute(2, 0, 1)


def generate_random_patch_center(
    img_shape: tuple[int, int] | np.ndarray,
    patch_shape: tuple[int, int] | np.ndarray,
    offset: int = 20,
) -> np.ndarray:
    """Randomly generate the patch center so patch+offset fits in the image.

    Args:
        img_shape: the input image shape (H, W).
        patch_shape: the shape of the patch to extract (H, W).
        offset: the minimum offset the patch must keep from the image borders.

    Returns:
        The coordinate of the center given as (y, x).
    """
    assert (np.array(img_shape) >= np.array(patch_shape) + 2 * offset).all(), (
        f"Assert error patch dimension {patch_shape} +2*offset {2 * offset} "
        f"is bigger than source image {img_shape}"
    )

    min_y = patch_shape[0] // 2 + offset
    max_y = img_shape[0] - patch_shape[0] // 2 - offset
    min_x = patch_shape[1] // 2 + offset
    max_x = img_shape[1] - patch_shape[1] // 2 - offset

    x_center = np.random.randint(min_x, max_x) if max_x > min_x else min_x

    y_center = np.random.randint(min_y, max_y) if max_y > min_y else min_y

    return np.array([x_center, y_center])


def cat_images(img0: Tensor, img1: Tensor, mode: str = "vertical") -> Tensor:
    """Concatenate two images, padding the smallest one.

    Args:
        img0: input image0
            [...xH0xW0]
        img1: input image1
            [...xH1xW1]
        mode: either <vertical> or <horizontal>.

    Returns:
        The concatenated image
            [...xH2xH3].
    """
    H0, W0 = img0.shape[-2:]
    H1, W1 = img1.shape[-2:]
    assert mode in [
        "vertical",
        "horizontal",
    ], "type not recognized, choose between <vertical> or <horizontal>"
    if mode == "vertical":
        W_max = max(W0, W1)
        img0_pad = pad_and_cut_image(img0, (H0, W_max))
        img1_pad = pad_and_cut_image(img1, (H1, W_max))
        img_cat = torch.cat((img0_pad, img1_pad), dim=-2)
    else:
        # horizontal
        H_max = max(H0, H1)
        img0_pad = pad_and_cut_image(img0, (H_max, W0))
        img1_pad = pad_and_cut_image(img1, (H_max, W1))
        img_cat = torch.cat((img0_pad, img1_pad), dim=-1)
    return img_cat
