"""MegaDepth/DISK dataset loader for descriptor training."""

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import imageio.v3 as io
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.dataset_paths import resolve_dataset_path
from utils.utils_3D import P_from_R_t, rotate_image_and_camera_z_axis

# Candidate dataset paths; override with the SANDESC_MEGADEPTH_PATH env var.
DATASET_CANDIDATES = [
    Path("/home/mattia/HDD_Fast/Megadepth/data"),  # local
    Path("/gpfs/data/fs72667/icgma_durso/Megadepth/data"),  # server
]


def rescale_and_pad(
    img: Tensor, depth: Tensor, K: Tensor, img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale the longest side to fit the image, and pad the other.

    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size.

    Returns:
        img: the rescaled and padded image
            C,H,W
        depth: the rescaled and padded depth map
            H,W
        K: the adapted intrinsics matrix
            3,3

    """
    H, W = img.shape[-2:]
    scale_factor = min(img_size / H, img_size / W)

    new_shape = (int(np.round(scale_factor * H)), int(np.round(scale_factor * W)))

    #  rescale
    img = F.interpolate(
        img[None],
        size=new_shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    depth = F.interpolate(depth[None, None], size=new_shape, mode="nearest")[0, 0]
    K[:2, :] *= scale_factor

    #  pad
    pad = (img_size - new_shape[1], img_size - new_shape[0])  # x,y
    img = F.pad(img, (0, pad[0], 0, pad[1]), mode="constant", value=0.0)
    depth = F.pad(depth, (0, pad[0], 0, pad[1]), mode="constant", value=float("nan"))
    return img, depth, K


def rescale_and_crop(
    img: Tensor, depth: Tensor, K: Tensor, img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale such that the shortest side fits the image, and crop the other.

    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size.

    Returns:
        img: the rescaled and cropped image
            C,H,W
        depth: the rescaled and cropped depth map
            H,W
        K: the adapted intrinsics matrix
            3,3

    """
    H, W = img.shape[-2:]
    scale_factor = max(img_size / H, img_size / W)

    new_shape = (int(np.round(scale_factor * H)), int(np.round(scale_factor * W)))

    #  rescale
    img = F.interpolate(
        img[None],
        size=new_shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    depth = F.interpolate(depth[None, None], size=new_shape, mode="nearest")[0, 0]
    K[:2, :] *= scale_factor

    #  crop
    img = img[:, :img_size, :img_size]
    depth = depth[:img_size, :img_size]
    return img, depth, K


class MegadepthDiskDataset(Dataset):
    """MegaDepth dataset using the DISK triplet split for training."""

    def __init__(
        self,
        img_size: int = 512,
        rescale_mode: str = "crop",
        random_rotation_degrees_fn: Callable | None = None,
        transform: transforms.Compose | None = None,
        verbose: bool = False,
    ) -> None:
        """Build the dataset.

        Args:
            img_size: the output image size.
            rescale_mode: how to rescale the images, either crop or pad.
            random_rotation_degrees_fn: a function returning a random rotation
                angle in degrees.
            transform: the transformation to apply to the images.
            verbose: whether to print messages when skipping invalid pairs.
        """
        if transform is None:
            transform = transforms.Compose([])
        assert rescale_mode in [
            "crop",
            "pad",
        ], "rescale_mode must be either crop or pad"

        self.img_size = img_size
        self.rescale_mode = rescale_mode
        self.verbose = verbose
        self.random_rotation_degrees_fn = random_rotation_degrees_fn
        self.transform = transforms.Compose(
            [t for t in transform.transforms if not isinstance(t, transforms.ToTensor)]
        )
        self.dataset_path = resolve_dataset_path(
            "megadepth-disk", "SANDESC_MEGADEPTH_PATH", DATASET_CANDIDATES
        )
        with (self.dataset_path / "dataset.json").open() as f:
            scenes = json.load(f)
        self.scenes = {k: scenes[k] for k in sorted(scenes.keys())}

    def reset(self) -> None:
        """Reset the dataloader."""
        self.__init__(
            img_size=self.img_size,
            rescale_mode=self.rescale_mode,
            random_rotation_degrees_fn=self.random_rotation_degrees_fn,
            transform=self.transform,
        )

    def __len__(self) -> int:
        """Return the dataset length (with margin for invalid images)."""
        #  here we should have 10000, but we keep some margin because we might
        #  have some invalid images
        return len(self.scenes) * 9000

    def __getitem__(self, idx: int) -> dict:
        """Get the pair of images.

        Args:
            idx: the index of the pair.

        Returns:
            A dictionary with the pair of images, the depth maps, and the
            camera intrinsics and extrinsics.
        """
        while True:
            current_scene_name = list(self.scenes.keys())[idx % len(self.scenes)]
            current_scene = self.scenes[current_scene_name]
            base_path = self.dataset_path / "scenes" / current_scene_name
            # Sample a triplet without replacement: pop() drains this worker's
            # in-memory copy of the scene's tuples over the epoch. reset()
            # reloads the full set from disk.
            triplets = current_scene["tuples"]
            triplet_idx = np.random.randint(len(triplets))

            idx0, idx1, idx2 = triplets.pop(triplet_idx)

            img0, depth0, K0, P0 = self.load_data(base_path, current_scene, idx0)
            img1, depth1, K1, P1 = self.load_data(base_path, current_scene, idx1)

            if img0 is None or img1 is None:
                if self.verbose:
                    print(
                        f"Skipping invalid pair {idx0}, {idx1} in scene "
                        f"{current_scene_name}"
                    )
                continue

            if self.random_rotation_degrees_fn is not None:
                angle1 = self.random_rotation_degrees_fn()
                img1, P1, K1, depth1 = rotate_image_and_camera_z_axis(
                    angle1, img1, P1, K1, depth1
                )

            if self.rescale_mode == "pad":
                #  DISK paper: longest edge to 768 and zero pad the rest, bilinear
                img0, depth0, K0 = rescale_and_pad(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_pad(img1, depth1, K1, self.img_size)
            else:
                #  ours: shortest edge to dim and crop the rest, nearest
                img0, depth0, K0 = rescale_and_crop(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_crop(img1, depth1, K1, self.img_size)

            depth0[depth0 == 0.0] = float("nan")
            depth1[depth1 == 0.0] = float("nan")

            return {
                "img0": self.transform(img0),
                "img1": self.transform(img1),
                "depth0": depth0,
                "depth1": depth1,
                "K0": K0,
                "K1": K1,
                "P0": P0,
                "P1": P1,
            }

    def load_data(
        self, base_path: Path, current_scene: dict, idx: int
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Load a single image with its depth map, intrinsics and pose.

        Args:
            base_path: path to the current scene directory.
            current_scene: the scene metadata dictionary.
            idx: the index of the image within the scene.

        Returns:
            The image, depth map, intrinsics K and pose P (or None on error).
        """
        img_path = base_path / "images" / current_scene["images"][idx]
        depth_path = base_path / "depth_maps" / f"{img_path.stem}.h5"
        calib_path = base_path / "calibration" / f"calibration_{img_path.name}.h5"

        try:
            img0 = torch.tensor(io.imread(img_path) / 255.0).permute(2, 0, 1).float()
            depth0 = torch.tensor(h5py.File(depth_path, "r")["depth"][()])
            calib_h5 = h5py.File(calib_path, "r")
            K, R, t = (
                torch.tensor(calib_h5["K"][()]),
                torch.tensor(calib_h5["R"][()]),
                torch.tensor(calib_h5["T"][()]),
            )
        except (OSError, KeyError, ValueError) as e:
            if self.verbose:
                print(f"Skipping {img_path.name}: {e}")
            return None, None, None, None

        P = P_from_R_t(R[None], t[None])[0]

        if (2 * K[0, 2]).round().int() != img0.shape[-1] or (
            2 * K[1, 2]
        ).round().int() != img0.shape[-2]:
            if self.verbose:
                print(f"{img_path.name} center is not centered with K, skipping")
            return None, None, None, None

        return img0, depth0, K, P
