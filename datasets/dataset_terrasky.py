"""TerraSky3D dataset loader."""

import json
from collections.abc import Callable
from pathlib import Path

import h5py
import imageio.v3 as io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.dataset_paths import resolve_dataset_path
from utils.utils_3D import rotate_image_and_camera_z_axis

# Candidate dataset paths; override with the SANDESC_TERRASKY_PATH env var.
DATASET_CANDIDATES = [
    Path("/home/mattia/Desktop/datasets/mydataset/data"),  # local
    # NOTE: server fallback below points at the Megadepth dir (pre-existing,
    # likely wrong for TerraSky); override with SANDESC_TERRASKY_PATH.
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


def rescale_and_center_crop(
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

    #  center crop
    # Calculate offsets for center crop
    y_offset = (img.shape[-2] - img_size) // 2
    x_offset = (img.shape[-1] - img_size) // 2
    img = img[:, y_offset : y_offset + img_size, x_offset : x_offset + img_size]
    depth = depth[y_offset : y_offset + img_size, x_offset : x_offset + img_size]
    # Update principal point
    K[0, 2] -= x_offset
    K[1, 2] -= y_offset
    return img, depth, K


class TerraSkyDataset(Dataset):
    """TerraSky3D dataset of cross-view (aerial/ground) image pairs."""

    def __init__(
        self,
        img_size: int = 512,
        rescale_mode: str = "crop",
        random_rotation_degrees_fn: Callable | None = None,
        transform: transforms.Compose | None = None,
        only_mixed: bool = False,  # not used
        verbose: bool = False,
    ) -> None:
        """Build the dataset.

        Args:
            img_size: the output image size.
            rescale_mode: how to rescale the images, either crop or pad.
            random_rotation_degrees_fn: a function returning a random rotation
                angle in degrees.
            transform: the transformation to apply to the images.
            only_mixed: keep only aerial/ground mixed pairs (currently unused).
            verbose: whether to print dataset statistics and skip messages.
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
            "TerraSky3D", "SANDESC_TERRASKY_PATH", DATASET_CANDIDATES
        )
        scenes = sorted(p.name for p in self.dataset_path.iterdir())

        with (self.dataset_path.parent / "train_data.json").open() as f:
            self.scenes = json.load(f)

        self.flattened_pairs = []
        bar = tqdm(scenes, desc="Loading scenes and pairs")
        for scene in bar:
            try:
                # this is one way to select pairs, but one can use whatever as
                # long as they are meaningful
                csv_name = (
                    "cyclic_depth_filtering_results1600_"
                    "bidirectionally_filtered_square.csv"
                )
                df = pd.read_csv(
                    self.dataset_path / scene / csv_name,
                    index_col=None,
                )
                # filter df by num_pixels > 3000 and th 10 > 0.5
                df = df[df["num_pixels"] > 3000]
                df = df[df["10px"] > 0.5]

            except FileNotFoundError:
                # this should not happen, but just in case we skip the scene if
                # the consistency check results are not found
                if self.verbose:
                    print(
                        f"Consistency check results not found for scene "
                        f"{scene}, skipping..."
                    )
                continue
            # ... here there might be some filtering of the pairs based on the
            # consistency check results
            pairs = df[["level_0", "level_1"]].values.tolist()

            if only_mixed:
                pairs = [
                    (img0, img1)
                    for img0, img1 in pairs
                    if ("aerial" in img0 and "aerial" not in img1)
                    or ("aerial" not in img0 and "aerial" in img1)
                ]

            if len(pairs) > 0:
                self.scenes[scene]["pairs"] = pairs

                for img0, img1 in pairs:
                    self.flattened_pairs.append((scene, (img0, img1)))

            else:
                # pop the scene if it has no valid pairs
                self.scenes.pop(scene)

        # count how many pair ahave "aerial" in one of the two images, or in both
        if self.verbose:
            mixed_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if ("aerial" in img0 and "aerial" not in img1)
                or ("aerial" not in img0 and "aerial" in img1)
            )
            aerial_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if "aerial" in img0 and "aerial" in img1
            )
            ground_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if "aerial" not in img0 and "aerial" not in img1
            )
            total_pairs = len(self.flattened_pairs)
            print(
                f"Mixed images:  {mixed_count:>10,} "
                f"({mixed_count / total_pairs:>7.2%})",
                f"Aerial images: {aerial_count:>10,} "
                f"({aerial_count / total_pairs:>7.2%})",
                f"Ground images: {ground_count:>10,} "
                f"({ground_count / total_pairs:>7.2%})",
                "-" * 40,
                f"Total pairs:   {total_pairs:>10,}",
                sep="\n",
            )

    def __len__(self) -> int:
        """Return the number of flattened image pairs."""
        return len(self.flattened_pairs)

    def __getitem__(self, idx: int) -> dict:
        """Get the pair of images.

        Args:
            idx: the index of the pair.

        Returns:
            A dictionary with the pair of images, the depth maps, and the
            camera intrinsics and extrinsics.
        """
        while True:
            # drawing a random scene
            current_scene_name = list(self.scenes.keys())[idx % len(self.scenes)]
            current_scene = self.scenes[current_scene_name]
            base_path = self.dataset_path / current_scene_name

            # drawing a pair of images from the current scene
            pairs = current_scene["pairs"]
            if pairs is None or len(pairs) == 0:
                if self.verbose:
                    print(f"No valid pairs in scene {current_scene_name}, skipping...")
                continue
            pair_idx = np.random.randint(len(pairs))
            img0_name, img1_name = pairs[pair_idx]

            img0, depth0, K0, P0 = self.load_data(base_path, img0_name, current_scene)
            img1, depth1, K1, P1 = self.load_data(base_path, img1_name, current_scene)

            if img0 is None or img1 is None:
                if self.verbose:
                    print(
                        f"Skipping invalid pair {img0_name}, {img1_name} in "
                        f"scene {current_scene_name}"
                    )
                continue

            if self.random_rotation_degrees_fn is not None:
                angle1 = self.random_rotation_degrees_fn()
                img1, P1, K1, depth1 = rotate_image_and_camera_z_axis(
                    angle1, img1, P1, K1, depth1
                )

            if self.rescale_mode == "pad":
                img0, depth0, K0 = rescale_and_pad(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_pad(img1, depth1, K1, self.img_size)
            else:
                img0, depth0, K0 = rescale_and_center_crop(
                    img0, depth0, K0, self.img_size
                )
                img1, depth1, K1 = rescale_and_center_crop(
                    img1, depth1, K1, self.img_size
                )

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
        self, base_path: Path, img: str, current_scene: dict
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Load a single image with its depth map, intrinsics and pose.

        Args:
            base_path: path to the current scene directory.
            img: the image file name.
            current_scene: the scene metadata dictionary.

        Returns:
            The image, depth map, intrinsics K and pose P.
        """
        img_path = base_path / "frames" / img
        depth_path = (
            str(img_path).replace("frames", "depth/maps").replace(".jpg", ".h5")
        )

        img_rgb = torch.from_numpy(io.imread(img_path) / 255.0).permute(2, 0, 1).float()
        with h5py.File(depth_path, "r") as f:
            depth = torch.from_numpy(f["depth"][()])

        img_entry = current_scene["images"][img]
        P = torch.tensor(img_entry["P"])
        P = torch.cat([P, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)

        # forcing the principal point to be in the center of the image, as we
        # will apply random crops and rotations; this might not be the best
        # choice but it works for now
        K = torch.tensor(current_scene["cameras"][str(img_entry["K_id"])]["K"])
        K[0, 2] = img_rgb.shape[-1] // 2
        K[1, 2] = img_rgb.shape[-2] // 2

        return img_rgb.float(), depth.float(), K.float(), P.float()

    def reset(self) -> None:
        """Reset the dataloader."""
        self.__init__(
            img_size=self.img_size,
            rescale_mode=self.rescale_mode,
            random_rotation_degrees_fn=self.random_rotation_degrees_fn,
            transform=self.transform,
        )
