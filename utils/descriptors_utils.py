from typing import Tuple, Callable

import gc
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import wandb
from utils.descriptor_stats import compute_stats
from utils.utils_logging import log_match_plot
from utils.helpers import seed_management


def extract_keypoints(
    img0: Tensor,
    img1: Tensor,
    detector_wrapper,
    max_n_keypoints: int,
    compute_stats_orig: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    B, C, H, W = img0.shape
    device = img0.device

    nkpts0_min = max_n_keypoints
    nkpts1_min = max_n_keypoints

    const = float("nan")
    # const = 1.0

    kpts0 = const * torch.zeros((B, max_n_keypoints, 2), device=device)  # B,n_kpts,2
    kpts1 = const * torch.zeros((B, max_n_keypoints, 2), device=device)  # B,n_kpts,2
    kpts0_scores = const * torch.zeros(
        (B, max_n_keypoints), device=device
    )  # B,n_kpts,2
    kpts1_scores = const * torch.zeros(
        (B, max_n_keypoints), device=device
    )  # B,n_kpts,2
    kpts0_scales = const * torch.zeros(
        (B, max_n_keypoints), device=device
    )  # B,n_kpts,2
    kpts1_scales = const * torch.zeros(
        (B, max_n_keypoints), device=device
    )  # B,n_kpts,2

    if compute_stats_orig:
        des0_orig = const * torch.zeros((B, max_n_keypoints, 128), device=device)
        des1_orig = const * torch.zeros((B, max_n_keypoints, 128), device=device)
    else:
        des0_orig, des1_orig = None, None

    for b in range(B):
        with torch.no_grad():
            output0_b = detector_wrapper.extract(img0[b], max_kpts=max_n_keypoints)
            output1_b = detector_wrapper.extract(img1[b], max_kpts=max_n_keypoints)
            kpts0_b = output0_b.kpts  # n_kpts0,2
            kpts1_b = output1_b.kpts  # n_kpts1,2
            kpts0_scores_b = output0_b.kpts_scores
            kpts1_scores_b = output1_b.kpts_scores
            kpts0_scales_b = output0_b.kpts_scales
            kpts1_scales_b = output1_b.kpts_scales
            des0_orig_b = output0_b.des  # n_kpts0,128
            des1_orig_b = output1_b.des  # n_kpts1,128
            nkpts0_min = min(nkpts0_min, kpts0_b.shape[0])
            nkpts1_min = min(nkpts1_min, kpts1_b.shape[0])

        kpts0[b, : kpts0_b.shape[0]] = kpts0_b
        kpts1[b, : kpts1_b.shape[0]] = kpts1_b
        kpts0_scores[b, : kpts0_b.shape[0]] = kpts0_scores_b
        kpts1_scores[b, : kpts1_b.shape[0]] = kpts1_scores_b
        kpts0_scales[b, : kpts0_b.shape[0]] = kpts0_scales_b
        kpts1_scales[b, : kpts1_b.shape[0]] = kpts1_scales_b
        if compute_stats_orig:
            des0_orig[b, : kpts0_b.shape[0]] = des0_orig_b
            des1_orig[b, : kpts1_b.shape[0]] = des1_orig_b

    return (
        kpts0,
        kpts1,
        kpts0_scores,
        kpts1_scores,
        kpts0_scales,
        kpts1_scales,
        des0_orig,
        des1_orig,
    )


@torch.no_grad()
def evaluate(
    detector_wrapper,
    network: nn.Module,
    valid_dataloader: DataLoader,
    compute_GT_matching_matrix_fn: Callable,
    current_interation: int,
    n_max_keypoints: int,
    n_iterations: int,
    n_plots: int,
    device: str,
    tag: str = "valid_",
    use_wrapper_descriptor: bool = False,
    compute_pose_stats: bool = False,
    ratio_test: float = 1.0,
    max_epipolar_error: float = 1.0,
    return_stats: bool = False,
    log_pca_feature_space=False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    seed_stored = seed_management("store_and_reset")

    if use_wrapper_descriptor is False:
        network_temp = deepcopy(network)
        network_temp.eval()
        network_temp.requires_grad_(False)
        detector_wrapper.add_custom_descriptor(
            network_temp
        )  # set the detector descriptor

    stats_all = []
    pose_error = []
    for i, data in tqdm(
        zip(range(n_iterations), valid_dataloader),
        position=1,
        desc="Evaluating",
        total=n_iterations,
    ):
        for key in [
            "img0",
            "img1",
            "H0_1",
            "depth0",
            "depth1",
            "P0",
            "P1",
            "K0",
            "K1",
            "embs0",
            "embs1",
        ]:
            if key in data:
                data[key] = data[key].to(device)

        img0 = data["img0"]  # B,C,H,W
        img1 = data["img1"]  # B,C,H,W

        if "embs0" in data:
            embs0 = data["embs0"]
            embs1 = data["embs1"]
            output0 = detector_wrapper.extract(
                img0[0], embs=embs0, max_kpts=n_max_keypoints
            )
            output1 = detector_wrapper.extract(
                img1[0], embs=embs1, max_kpts=n_max_keypoints
            )
        else:
            output0 = detector_wrapper.extract(img0[0], max_kpts=n_max_keypoints)
            output1 = detector_wrapper.extract(img1[0], max_kpts=n_max_keypoints)
        kpts0 = output0.kpts[None]  # 1,n_kpts0,2
        kpts1 = output1.kpts[None]  # 1,n_kpts1,2
        des0 = output0.des[None]
        des1 = output1.des[None]

        data["kpts0"] = kpts0
        data["kpts1"] = kpts1
        score_matrix = des0 @ des1.permute(
            0, 2, 1
        )  # 1,n_kpts0,n_kpts1 | Cosine Similarity matrix in [-1,+1]
        # < ==========================================================================================================
        if compute_pose_stats:
            # Pose estimation
            from utils.utils_matches import MNN
            from mylib import metrics, geometry

            # MNN matching
            matcher = MNN(min_score=0.5, ratio_test=ratio_test, device=device)
            matches = matcher.match([des0[0]], [des1[0]])[0].matches

            kpts0_matched = kpts0[0][matches[:, 0]].cpu()
            kpts1_matched = kpts1[0][matches[:, 1]].cpu()
            K0 = data["K0"][0].cpu().numpy()
            K1 = data["K1"][0].cpu().numpy()
            camera_dict0 = {
                "model": "SIMPLE_PINHOLE",
                "width": img0.shape[-1],
                "height": img0.shape[-2],
                "params": np.array([K0[0, 0], K0[0, 2], K0[1, 2]]),
            }
            camera_dict1 = {
                "model": "SIMPLE_PINHOLE",
                "width": img1.shape[-1],
                "height": img1.shape[-2],
                "params": np.array([K1[0, 0], K1[0, 2], K1[1, 2]]),
            }
            _, _, dR, dt = geometry.compute_essential_poselib(
                kpts0_matched,
                kpts1_matched,
                camera_dict0,
                camera_dict1,
                return_Rt=True,
                max_epipolar_error=max_epipolar_error,
            )
            dR_gt, dt_gt = geometry.compute_relative_camera_motion(
                P1=data["P0"], P2=data["P1"]
            )
            err = metrics.evaluate_R_t(dR_gt[0], dt_gt, dR, dt[None])

            pose_error.append(err)
        # < ============================================================================================================

        # compute GT the matches
        matches_matrix_GT_with_bins = compute_GT_matching_matrix_fn(
            data
        )  # B,n_kpts0+1,n_kpts1+1
        stats, _ = compute_stats(
            score_matrix, matches_matrix_GT_with_bins, ratio_test=ratio_test
        )
        stats["n_kpts"] = 0.5 * (kpts0.shape[1] + kpts1.shape[1])
        stats["n_training_matches"] = (
            matches_matrix_GT_with_bins[:, :-1, :-1].sum().cpu().item()
        )
        stats["n_training_bin_matches"] = (
            (
                matches_matrix_GT_with_bins[:, :, -1].sum()
                + matches_matrix_GT_with_bins[:, -1, :].sum()
            )
            .cpu()
            .item()
        )
        stats_all += [stats]

        caption_keys = {
            "n_matches_correct": "4d",
            "avg_margin_correct": ".2f",
            "n_matches_mismatched": "4d",
            "avg_margin_mismatched": ".2f",
            "n_matches_inexistent": "4d",
            "avg_margin_inexistent": ".2f",
            "matches_precision": ".2f",
            "matches_recall": ".2f",
        }
        caption = ""
        for j in range(len(caption_keys) // 2):
            key0, key1 = (
                list(caption_keys.keys())[2 * j],
                list(caption_keys.keys())[2 * j + 1],
            )
            caption += (
                f"{key0[:14]}: {stats[key0]:{caption_keys[key0]}}   "
                f"{key1[:13]}: {stats[key1]:{caption_keys[key1]}}\n"
            )
        caption = (
            f"n_matches_GT: {matches_matrix_GT_with_bins[:, :-1, :-1].sum():.0f}\n"
            f"{caption}"
        )

        if i < n_plots and wandb.run is not None:
            from datasets.dataset_loaders import (
                transform_from_normalized_rgb_to_grayscale,
            )

            if img0.shape[1] == 3:
                img0 = transform_from_normalized_rgb_to_grayscale(img0)
                img1 = transform_from_normalized_rgb_to_grayscale(img1)
            log_match_plot(
                img0,
                img1,
                kpts0,
                kpts1,
                score_matrix,
                matches_matrix_GT_with_bins,
                batch_idx=i,
                iteration=current_interation,
                tag=tag,
                caption=caption,
            )

        del (
            data,
            output0,
            output1,
            des0,
            des1,
            kpts0,
            kpts1,
            score_matrix,
            matches_matrix_GT_with_bins,
            img0,
            img1,
            kpts0_matched,
            kpts1_matched,
        )
        gc.collect()
        torch.cuda.empty_cache()

    stats_df = pd.DataFrame(stats_all)
    stats_mean_df = stats_df.mean(axis=0)
    stats_mean_df = stats_mean_df.add_prefix(tag)

    # convert pandas to dict
    stats_mean = stats_mean_df.to_dict()

    if compute_pose_stats:
        # pose_stats
        pose_error = np.stack(pose_error)
        # to pandas
        pose_error_df = pd.DataFrame(pose_error, columns=["err_R", "err_t"])
        # add column with max of both
        pose_error_df["max"] = pose_error_df[["err_R", "err_t"]].max(axis=1)

        # compute AUC
        auc_max = metrics.compute_AUC_pxsfm(
            pose_error_df["max"].values, thresholds=[1, 3, 5, 10], min_error=None
        )
        auc_R = metrics.compute_AUC_pxsfm(
            pose_error_df["err_R"].values, thresholds=[1, 3, 5, 10], min_error=None
        )
        auc_t = metrics.compute_AUC_pxsfm(
            pose_error_df["err_t"].values, thresholds=[1, 3, 5, 10], min_error=None
        )

        stats_mean[f"{tag}pose_auc_max@1"] = auc_max[0]
        stats_mean[f"{tag}pose_auc_max@3"] = auc_max[1]
        stats_mean[f"{tag}pose_auc_max@5"] = auc_max[2]
        stats_mean[f"{tag}pose_auc_max@10"] = auc_max[3]
        # stats_mean[f'{tag}pose_auc_R@1'] = auc_R[0]
        # stats_mean[f'{tag}pose_auc_R@3'] = auc_R[1]
        # stats_mean[f'{tag}pose_auc_R@5'] = auc_R[2]
        # stats_mean[f'{tag}pose_auc_R@10'] = auc_R[3]
        # stats_mean[f'{tag}pose_auc_t@1'] = auc_t[0]
        # stats_mean[f'{tag}pose_auc_t@3'] = auc_t[1]
        # stats_mean[f'{tag}pose_auc_t@5'] = auc_t[2]
        # stats_mean[f'{tag}pose_auc_t@10'] = auc_t[3]

    if return_stats:
        return stats_df, pose_error_df if compute_pose_stats else None
    else:
        if wandb.run is not None:
            wandb.log(stats_mean, step=current_interation)

        # if use_wrapper_descriptor is False:
        #     network.requires_grad_(True)
        #     network.train()
        #     detector_wrapper.custom_descriptors = None

        seed_management("restore", seed_stored)

    del (
        stats_df,
        stats_mean_df,
        stats_mean,
        pose_error_df,
        pose_error,
        auc_max,
        auc_R,
        auc_t,
        detector_wrapper.custom_descriptor,
    )
    gc.collect()
    torch.cuda.empty_cache()


def create_fake_score_matrix_from_matched_ktps(kps1, kps2, idxs, device="cuda"):
    """
    Given two sets of matched keypoints, create a score matrix.
    # not working for batch != 1
    """
    m = torch.zeros((1, kps1.shape[0], kps2.shape[0]), device=device)
    m[0, idxs[:, 0], idxs[:, 1]] = 1.0

    return m
