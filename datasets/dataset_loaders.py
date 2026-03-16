from typing import Tuple, Dict, Callable, Union

import numpy as np
import torch
from torch import Tensor

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.v2 as v2

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from datasets.dataset_imb import ImageMatchingBenchmark as IMBDataset  # Use alias
from datasets.dataset_megadepth_disk import MegadepthDiskDataset
from datasets.dataset_terrasky import TerraSkyDataset

from utils.utils_3D import compute_GT_matches_matrix_3D

transform_from_normalized_rgb_to_grayscale = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        transforms.Grayscale(),
        transforms.Normalize(mean=0.5, std=0.5),
    ]
)


def compute_GT_matching_matrix_3D(
    data: Dict[str, Tensor],
    return_distances_and_projected: bool = False,
    allow_multiple_matches: bool = False,
) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
    """wrapper for the function compute_GT_matches_matrix_3D"""

    if return_distances_and_projected:
        GT_matrix, xy0_proj, xy1_proj, dist0, dist1 = compute_GT_matches_matrix_3D(
            data["kpts0"],
            data["kpts1"],
            data["depth0"],
            data["depth1"],
            data["P0"],
            data["P1"],
            data["K0"],
            data["K1"],
            max_relative_depth_error=0.1,
            max_pixel_error=3.0,
            min_pixel_error_for_unmatched=5.0,
            mode="nearest",
            return_distances_and_projected=True,
            allow_multiple_matches=allow_multiple_matches,
        )
        return GT_matrix, xy0_proj, xy1_proj, dist0, dist1

    GT_matrix = compute_GT_matches_matrix_3D(
        data["kpts0"],
        data["kpts1"],
        data["depth0"],
        data["depth1"],
        data["P0"],
        data["P1"],
        data["K0"],
        data["K1"],
        max_relative_depth_error=0.1,
        max_pixel_error=3.0,
        min_pixel_error_for_unmatched=5.0,
        mode="nearest",
        return_distances_and_projected=False,
        allow_multiple_matches=allow_multiple_matches,
    )
    return GT_matrix


def get_dataloaders(
    mode: str,
    batch_size: int,
    img_channels: int,
    num_workers: int = 0,
    config_override: Dict = None,
    augment: bool = False,
    img_size: int = 512,
) -> Tuple[DataLoader, DataLoader, Callable, Dict[str, float]]:
    """Get the dataloaders for the specified mode

    Args:
        mode: which dataset to load
        batch_size: the batch size
        img_channels: the number of channels of the images
        num_workers: the number of workers for the dataloaders
        config_override: the configuration override
    Returns:
        the training and evaluation dataloaders, the function to compute the GT matching matrix and the configuration of the dataset
    """

    config_override = {} if config_override is None else config_override
    if img_channels == 1:
        img_norm = [transforms.Grayscale(), transforms.Normalize(mean=0.5, std=0.5)]
    elif img_channels == 3:
        img_norm = [
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # let's stay in the 0-1 range to avoid problem in plots
        ]
    else:
        raise ValueError("img_channels must be 1 or 3")

    if mode == "imb":
        compute_GT_matching_matrix_fn = compute_GT_matching_matrix_3D
        transform_imb = transforms.Compose([transforms.ToTensor(), *img_norm])
        config_imb = {
            "img_size": (img_size, img_size),  # x,y
            "scenes": [
                "reichstag",
                "sacre_coeur",
                "st_peters_square",
            ],  # IMB validation scenes
            "covisibility_weights": {
                "0.0": 0.0,
                "0.1": 0.1,
                "0.2": 0.1,
                "0.3": 0.1,
                "0.4": 0.1,
                "0.5": 0.1,
                "0.6": 0.1,
                "0.7": 0.1,
                "0.8": 0.1,
                "0.9": 0.0,
            },
        }
        if config_override is not None:
            config_imb = {**config_imb, **config_override}
        dataset_training = IMBDataset(
            config_imb["covisibility_weights"],
            scenes=config_imb["scenes"],
            img_shape=(config_imb["img_size"][1], config_imb["img_size"][0]),
            transform=transform_imb,
        )
        dataset_validation = IMBDataset(
            config_imb["covisibility_weights"],
            scenes=config_imb["scenes"],
            # scenes=['reichstag', 'sacre_coeur', 'st_peters_square'],
            img_shape=(config_imb["img_size"][1], config_imb["img_size"][0]),
            # img_shape=None,
            transform=transform_imb,
        )
        config_dataset = config_imb
        config_dataset["dataset"] = "imb"

    elif mode == "disk":
        compute_GT_matching_matrix_fn = compute_GT_matching_matrix_3D
        if augment and img_channels == 3:
            transform_disk = v2.Compose(
                [
                    *img_norm,
                    v2.RandomPhotometricDistort(
                        brightness=(0.75, 1.25),  # ±25% brightness variation
                        contrast=(0.4, 1.6),  # ±60% contrast variation
                        saturation=(0.4, 1.6),  # ±60% saturation variation
                        hue=(-0.08, 0.08),  # ±8% hue variation
                        p=0.5,  # 50% probability of applying each distortion
                    ),
                    v2.RandomApply(
                        torch.nn.ModuleList(
                            [
                                v2.GaussianNoise(
                                    mean=0.0,
                                    sigma=0.2,
                                    clip=True,  # Ensure values remain within [0, 1]
                                ),
                                v2.GaussianBlur(
                                    kernel_size=7,  # 7x7 kernel
                                    sigma=(0.2, 3.0),  # Sigma range for blur
                                ),
                            ]
                        ),
                        p=0.3,  # 30% probability of applying noise or blur
                    ),
                ]
            )
        else:
            transform_disk = transforms.Compose([*img_norm])
        config_disk = {
            "img_size": img_size,  # x,y
            "rescale_mode": "crop",
            # 'rescale_mode': 'pad',
        }
        if config_override is not None:
            config_disk = {**config_disk, **config_override}
        dataset_training = MegadepthDiskDataset(**config_disk, transform=transform_disk)
        dataset_validation = MegadepthDiskDataset(
            **config_disk, transform=transform_disk
        )
        config_dataset = config_disk
        config_dataset["dataset"] = "disk"

    elif mode == "terrasky":
        compute_GT_matching_matrix_fn = compute_GT_matching_matrix_3D
        transform_terrasky = transforms.Compose([transforms.ToTensor(), *img_norm])
        config_terrasky = {
            "img_size": img_size,  # x,y
            "rescale_mode": "crop",
        }
        if config_override is not None:
            config_terrasky = {**config_terrasky, **config_override}
        dataset_training = TerraSkyDataset(
            **config_terrasky, transform=transform_terrasky
        )
        dataset_validation = TerraSkyDataset(
            **config_terrasky, transform=transform_terrasky
        )
        config_dataset = config_terrasky
        config_dataset["dataset"] = "terrasky"
    else:
        raise ValueError("dataset mode not recognized")

    #  set to each worker a different seed, but in a way that every time the code is run the seed is the same
    dataloader_train = DataLoader(
        dataset_training,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
        worker_init_fn=lambda x: np.random.seed(torch.initial_seed() % (2**32 - 1)),
        pin_memory=True,
    )
    dataloader_eval = DataLoader(
        dataset_validation,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        shuffle=False,
        worker_init_fn=lambda _: np.random.seed(0),
        pin_memory=False,
    )

    return (
        dataloader_train,
        dataloader_eval,
        compute_GT_matching_matrix_fn,
        config_dataset,
    )
