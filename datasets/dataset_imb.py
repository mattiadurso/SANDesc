"""Image Matching Benchmark (IMB) dataset, used for validation."""

from pathlib import Path

import h5py
import imageio.v3 as io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.dataset_paths import resolve_dataset_path
from utils.utils_3D import P_from_R_t_np, scale_and_crop

# Candidate dataset paths; override with the SANDESC_IMB_PATH env var.
DATASET_CANDIDATES = [
    Path("/home/mattia/HDD_Fast/Datasets/IMB/validation"),  # local
    Path("/gpfs/data/fs72667/icgma_durso/IMB/validation"),  # server
]

COVISIBILITY_THRS = [
    "0.0",
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
]


class ImageMatchingBenchmark(Dataset):
    """IMB validation dataset of covisible image pairs with depth and pose."""

    def __init__(
        self,
        covisibility_weights: dict[str, float],
        scenes: list[str],
        img_shape: tuple[int, int] | None,
        transform: transforms.Compose | None = None,
    ) -> None:
        """Image Matching Benchmark dataset.

        Args:
            covisibility_weights: fraction of pairs to use for each
                covisibility threshold.
            scenes: the scenes to use.
            img_shape: the output image shape.
            transform: the transform to apply to the images.
        """
        if transform is None:
            transform = transforms.Compose([])
        self.dataset_path = resolve_dataset_path(
            "IMB", "SANDESC_IMB_PATH", DATASET_CANDIDATES
        )
        assert sorted(covisibility_weights.keys()) == COVISIBILITY_THRS
        self.covisibility_probs = np.array(list(covisibility_weights.values()))
        self.covisibility_probs /= self.covisibility_probs.sum()
        self.transform = transform
        self.scenes = scenes
        self.img_shape = img_shape

        self.covisibilities = {}
        #  load the covisibility values
        for scene in self.scenes:
            self.covisibilities[scene] = {}
            base_path = self.dataset_path / scene / "set_100" / "new-vis-pairs"
            for i in range(9, -1, -1):
                keys_file_name = f"keys-th-{i / 10:.1f}.npy"
                self.covisibilities[scene][f"{i / 10:.1f}"] = list(
                    np.load(str(base_path / keys_file_name))
                )
        #  remove every pair that appears in a lower covisibility from all
        #  the higher ones
        for scene in self.scenes:
            for i in range(0, 10, 1):
                for j in range(i + 1, 10, 1):
                    self.covisibilities[scene][f"{i / 10:.1f}"] = sorted(
                        list(
                            set(self.covisibilities[scene][f"{i / 10:.1f}"])
                            - set(self.covisibilities[scene][f"{j / 10:.1f}"])
                        )
                    )

        #  load all the images
        self.images = {}

    def __len__(self) -> int:
        """Return the total number of covisible image pairs."""
        n_elements = 0
        for scene in self.scenes:
            for covisibility in COVISIBILITY_THRS:
                n_elements += len(self.covisibilities[scene][covisibility])
        return n_elements

    def __getitem__(self, idx: int) -> dict:
        """Return a random covisible image pair with depth, intrinsics, pose."""
        current_scene = self.scenes[idx % len(self.scenes)]
        while True:
            #  here we loop because some images in the dataset are very small,
            #  and if we pick one of those, we raise an error and pick another
            #  pair
            try:
                while True:
                    #  loop until we find a covisibility where there is at
                    #  least one image pair
                    covisibility = np.random.choice(
                        COVISIBILITY_THRS, p=self.covisibility_probs
                    )
                    possible_pairs = self.covisibilities[current_scene][covisibility]
                    if possible_pairs:
                        break
                pair = np.random.choice(possible_pairs)
                img0_name, img1_name = pair.split("-")

                base_path = self.dataset_path / current_scene / "set_100"

                img0_path = (base_path / "images" / img0_name).with_suffix(".jpg")
                img1_path = (base_path / "images" / img1_name).with_suffix(".jpg")

                img0 = io.imread(img0_path)
                img1 = io.imread(img1_path)

                depth0_h5 = h5py.File(
                    (base_path / "depth_maps" / img0_name).with_suffix(".h5"), "r"
                )
                depth0 = depth0_h5["depth"][()]
                depth0 = depth0.astype(np.float32)
                depth0[depth0 == 0] = float("nan")
                depth1_h5 = h5py.File(
                    (base_path / "depth_maps" / img1_name).with_suffix(".h5"), "r"
                )
                depth1 = depth1_h5["depth"][()]
                depth1 = depth1.astype(np.float32)
                depth1[depth1 == 0] = float("nan")

                #  for some reasons, some images have a different shape than
                #  the depth map
                if img0.shape[:2] != depth0.shape:
                    min_shape = np.minimum(img0.shape[:2], depth0.shape)
                    img0 = img0[: min_shape[0], : min_shape[1]]
                    depth0 = depth0[: min_shape[0], : min_shape[1]]
                if img1.shape[:2] != depth1.shape:
                    min_shape = np.minimum(img1.shape[:2], depth1.shape)
                    img1 = img1[: min_shape[0], : min_shape[1]]
                    depth1 = depth1[: min_shape[0], : min_shape[1]]

                calibration0_h5 = h5py.File(
                    (
                        base_path / "calibration" / f"calibration_{img0_name}"
                    ).with_suffix(".h5"),
                    "r",
                )
                K0 = np.array(calibration0_h5["K"][()])
                R0 = np.array(calibration0_h5["R"][()])
                T0 = np.array(calibration0_h5["T"][()])
                P0 = P_from_R_t_np(R0, T0)
                calibration1_h5 = h5py.File(
                    (
                        base_path / "calibration" / f"calibration_{img1_name}"
                    ).with_suffix(".h5"),
                    "r",
                )
                K1 = np.array(calibration1_h5["K"][()])
                R1 = np.array(calibration1_h5["R"][()])
                T1 = np.array(calibration1_h5["T"][()])
                P1 = P_from_R_t_np(R1, T1)

                if self.img_shape is not None:
                    #  find a random point such that the img_shape is fully
                    #  contained in the image
                    center0 = np.array([img0.shape[1], img0.shape[0]]) // 2  # x,y
                    center1 = np.array([img1.shape[1], img1.shape[0]]) // 2  # x,y
                    img0, K0, _, bbox0, depth0 = scale_and_crop(
                        img0,
                        K0,
                        self.img_shape,
                        depth=depth0,
                        center=center0,
                        max_random_offset=999999,
                    )
                    img1, K1, _, bbox1, depth1 = scale_and_crop(
                        img1,
                        K1,
                        self.img_shape,
                        depth=depth1,
                        center=center1,
                        max_random_offset=999999,
                    )

                return {
                    "img0": self.transform(img0.copy()),
                    "img1": self.transform(img1.copy()),
                    "depth0": depth0,
                    "depth1": depth1,
                    "K0": K0,
                    "K1": K1,
                    "P0": P0,
                    "P1": P1,
                }

            except Exception:
                continue
