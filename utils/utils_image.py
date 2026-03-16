import torchvision.transforms.functional
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, Union


def crop_multiple_of(img: Tensor, multiple: int) -> Tensor:
    """crop an image from the input starting from the top-left corner such that both crop_H and crop_W are multiple of
    the provided number. crop_H and crop_W can be different, and are chosen as the closest to the original resolution
    Args:
        img: input image tensor
            ...,H,W
        multiple: output dimension will be multiple of this number
    Returns:
        Tensor: the output crop
    """
    H, W = img.shape[-2:]

    H_crop = (H // multiple) * multiple
    W_crop = (W // multiple) * multiple

    img_out = img[..., :H_crop, :W_crop]
    return img_out


def denormalize(
    img: Tensor, RGB_mean=(0.485, 0.456, 0.406), RGB_std=(0.229, 0.224, 0.225)
) -> Tensor:
    """denormalize the image color given the parameters used for the normalization
    Args:
        img: input image
            BxCxHxW or BxHxW or CxHxW or HxW
        RGB_mean: mean values used for the normalization
        RGB_std: std values used for the normalization
    Returns:
        output_img: the rescaled color image
    Raises:
        None
    """
    output_img = torchvision.transforms.functional.normalize(
        img,
        mean=[
            -RGB_mean[0] / RGB_std[0],
            -RGB_mean[1] / RGB_std[1],
            -RGB_mean[2] / RGB_std[2],
        ],
        std=[1 / RGB_std[0], 1 / RGB_std[1], 1 / RGB_std[2]],
    )

    return output_img


def pad_and_cut_image(
    img: Tensor,
    shape: Tuple[int, int],
    mode: str = "center",
    value: float = float("nan"),
    allow_cuts: bool = False,
) -> Tensor:
    """pad the input image such that the output shape is as required, the image is kept in the center
    Args:
        img: the input image
            [... x H x W]
        shape: the required shape as (H, W)
        value: the value used for the padding
        mode: the padding mode, 'center' or 'bottom-right'
        allow_cuts: whether to allow cutting the image or not
    Returns:
        Tensor: the padded image
    """
    H, W = img.shape[-2:]
    assert mode in ["center", "bottom-right"]
    if allow_cuts:
        img = img.clone()[..., : min(shape[0], H), : min(shape[1], W)]
        H, W = img.shape[-2:]
    else:
        assert H <= shape[0] and W <= shape[1]

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

    img_padded = F.pad(img, [pad_l, pad_r, pad_t, pad_b], value=value)

    return img_padded


def gray_to_colormap(
    img: Tensor, cmap: str = "inferno", rescale: bool = True
) -> Tensor:
    """Convert a grayscale images to a specific matplotlib colormap
    Args:
        img: the input batched grayscale images
            HxW or 1xHxW
        cmap: string that refer to a matplotlib colormap
        rescale: whether to rescale the colors
    Returns:
        Tensor: the output RGB image
            3xHxW
    Raises:
        None
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
    img_shape: Union[Tuple[int, int], np.ndarray],
    patch_shape: Union[Tuple[int, int], np.ndarray],
    offset: int = 20,
) -> np.ndarray:
    """randomly generate the central point of the patch in a way that the patch+offset is completely contained in
    the image
    Args:
        img_shape: the input image shape (H, W)
        patch_shape: the shape of the patch to extract (H, W)
        offset: the minimum offset that the patch must have with respect to the image borders
    Returns:
        array: the coordinate of the center given as (y, x)
    """
    assert (
        np.array(img_shape) >= np.array(patch_shape) + 2 * offset
    ).all(), f"Assert error patch dimension {patch_shape} +2*offset {2 * offset}  is bigger than source image {img_shape}"

    min_y = patch_shape[0] // 2 + offset
    max_y = img_shape[0] - patch_shape[0] // 2 - offset
    min_x = patch_shape[1] // 2 + offset
    max_x = img_shape[1] - patch_shape[1] // 2 - offset

    if max_x > min_x:
        x_center = np.random.randint(min_x, max_x)
    else:
        x_center = min_x

    if max_y > min_y:
        y_center = np.random.randint(min_y, max_y)
    else:
        y_center = min_y

    return np.array([x_center, y_center])


def cat_images(img0: Tensor, img1: Tensor, mode: str = "vertical") -> Tensor:
    """concatenate two images padding the smallest one
    Args:
        img0: input image0
            [...xH0xW0]
        img1: input image1
            [...xH1xW1]
        mode: either <vertical> or <horizontal>
    Returns:
        img_cat: the concatenated image
            [...xH2xH3]
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
