from pathlib import Path

import json
import h5py
import imageio.v3 as io
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_3D import P_from_R_t, rotate_image_and_camera_z_axis

# Define primary and fallback dataset paths
primary_path = Path("/home/mattia/HDD_Fast/Megadepth/data")  # local
fallback_path = Path("/gpfs/data/fs72667/icgma_durso/Megadepth/data")  # server

# Select the first available dataset path
if primary_path.exists():
    DATASET_PATH = primary_path
elif fallback_path.exists():
    DATASET_PATH = fallback_path
else:
    exit("Dataset megadepth-disk not found")


def rescale_and_pad(
    img: Tensor, depth: Tensor, K: Tensor, img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale the longest side to fit the image, and pad the other
    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size

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
    """Scale such that the shortest side fit the image, and crop the other
    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size

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
    def __init__(
        self,
        img_size: int = 512,
        rescale_mode: str = "crop",
        random_rotation_degrees_fn: callable = None,
        transform: transforms.Compose = transforms.Compose([]),
        verbose: bool = False,
    ):
        """
        Args:
            img_size: the output image size
            rescale_mode: how to rescale the images, either crop or pad
            random_rotation_degrees_fn: a function that returns a random rotation angle in degrees
            transform: the transformation to apply to the images
        """
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
        scenes = json.load(open(DATASET_PATH / "dataset.json"))
        self.scenes = {k: scenes[k] for k in sorted(scenes.keys())}

    def reset(self):
        """reset the dataloader"""
        self.__init__(
            img_size=self.img_size,
            rescale_mode=self.rescale_mode,
            random_rotation_degrees_fn=self.random_rotation_degrees_fn,
            transform=self.transform,
        )

    def __len__(self):
        #  here we should have 10000, but we keep some margin because we might have some invalid images
        return len(self.scenes) * 9000

    def __getitem__(self, idx: int) -> dict:
        """get the pair of images
        Args:
            idx: the index of the pair

        Returns:
            output: a dictionary containing the pair of images, the depth maps, the camera intrinsics and extrinsics
        """

        while True:
            current_scene_name = list(self.scenes.keys())[idx % len(self.scenes)]
            current_scene = self.scenes[current_scene_name]
            base_path = DATASET_PATH / "scenes" / current_scene_name
            triplets = current_scene["tuples"]
            triplet_idx = np.random.randint(len(triplets))

            idx0, idx1, idx2 = triplets.pop(triplet_idx)

            img0, depth0, K0, P0 = self.load_data(base_path, current_scene, idx0)
            img1, depth1, K1, P1 = self.load_data(base_path, current_scene, idx1)
            # img2, depth2, K2, P2 = self.load_data(base_path, current_scene, idx2)

            if img0 is None or img1 is None:  # or img2 is None:
                if self.verbose:
                    print(
                        f"Skipping invalid pair {idx0}, {idx1} in scene {current_scene_name}"
                    )
                continue

            if self.random_rotation_degrees_fn is not None:
                angle1 = self.random_rotation_degrees_fn()
                img1, P1, K1, depth1 = rotate_image_and_camera_z_axis(
                    angle1, img1, P1, K1, depth1
                )
                # angle2 = self.random_rotation_degrees_fn()
                # img2, P2, K2, depth2 = rotate_image_and_camera_z_axis(angle2, img2, P2, K2, depth2)

            if self.rescale_mode == "pad":
                #  DISK paper: longest edge to 768 and zero pad the rest, bilinear
                img0, depth0, K0 = rescale_and_pad(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_pad(img1, depth1, K1, self.img_size)
                # img2, depth2, K2 = rescale_and_pad(img2, depth2, K2, self.img_size)
            else:
                #  ours: shortest edge to dim and crop the rest, nearest
                img0, depth0, K0 = rescale_and_crop(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_crop(img1, depth1, K1, self.img_size)
                # img2, depth2, K2 = rescale_and_crop(img2, depth2, K2, self.img_size)

            depth0[depth0 == 0.0] = float("nan")
            depth1[depth1 == 0.0] = float("nan")
            # depth2[depth2 == 0.0] = float('nan')

            output = {
                # "scene": current_scene_name,
                "img0": self.transform(img0),
                "img1": self.transform(img1),
                # 'img2': self.transform(img2),
                "depth0": depth0,
                "depth1": depth1,
                # 'depth2': depth2,
                "K0": K0,
                "K1": K1,
                # 'K2': K2,
                "P0": P0,
                "P1": P1,
                # 'P2': P2,
            }
            return output

    def load_data(self, base_path, current_scene, idx: int):
        """load the pair of images
        Args:
            idx: the index of the pair

        Returns:
            output: a dictionary containing the pair of images, the depth maps, the camera intrinsics and extrinsics
        """
        img_path = base_path / "images" / current_scene["images"][idx]
        depth_path = base_path / "depth_maps" / f"{img_path.stem}.h5"
        calib_path = base_path / "calibration" / f"calibration_{img_path.name}.h5"

        img0 = torch.tensor(io.imread(img_path) / 255.0).permute(2, 0, 1).float()
        depth0 = torch.tensor(h5py.File(depth_path, "r")["depth"][()])
        calib_h5 = h5py.File(calib_path, "r")
        K, R, t = (
            torch.tensor(calib_h5["K"][()]),
            torch.tensor(calib_h5["R"][()]),
            torch.tensor(calib_h5["T"][()]),
        )
        P = P_from_R_t(R[None], t[None])[0]

        if (2 * K[0, 2]).round().int() != img0.shape[-1] or (
            2 * K[1, 2]
        ).round().int() != img0.shape[-2]:
            print("img0 center is not centered with the intrinsics, skipping this pair")
            return None, None, None, None

        return img0, depth0, K, P
