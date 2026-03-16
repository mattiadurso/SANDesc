from typing import Dict, Tuple

from torch import Tensor
import torch

from utils.utils_2D import (
    add_bins_to_matching_matrix,
    mutual_nearest_neighbors_from_score_matrix,
    compute_correct_wrong_mismatched_inexistent_unsure_matches,
)


def get_margin_and_ratio_from_scores_and_mnn_matrix(
    mnn_matrix: Tensor,
    best_scores0: Tensor,
    second_best_scores0: Tensor,
    second_best_scores1: Tensor,
) -> Tuple[Tensor, Tensor]:
    batch_matches, rows_matches, column_matches = torch.where(
        mnn_matrix
    )  # (n_matches), (n_matches), (n_matches)
    best_scores0_matches = best_scores0[
        batch_matches, rows_matches
    ]  # n_matches_proposed
    # by definition of mnn, the best_scores0_matches are exactly the same as best_scores0_matches
    # best_scores1_matches = best_scores1[batch_matches, column_matches]  # n_matches_proposed
    second_best_scores0_matches = second_best_scores0[
        batch_matches, rows_matches
    ]  # n_matches_proposed
    second_best_scores1_matches = second_best_scores1[
        batch_matches, column_matches
    ]  # n_matches_proposed
    margin = best_scores0_matches - torch.max(
        second_best_scores0_matches, second_best_scores1_matches
    )  # n_matches_proposed
    ratio = (
        torch.max(second_best_scores0_matches, second_best_scores1_matches)
        / best_scores0_matches
    )  # n_matches_proposed
    return margin, ratio


@torch.no_grad()
def compute_stats(
    score_matrix_des: Tensor,
    matching_matrix_GT_with_bins: Tensor,
    min_score: float = -1.0,
    ratio_test: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
    """
    Args:
        score_matrix_des:
            B,n_kpts0,n_kpts1
        matching_matrix_GT_with_bins:
            B,n_kpts0+1,n_kpts1+1
        min_score: the minimum score to consider a match
        ratio_test: the ratio test to consider a match
    """
    GT_matching_mask = matching_matrix_GT_with_bins[:, :-1, :-1]
    B, n0, n1 = GT_matching_mask.shape

    matching_matrix_GT_with_bins = add_bins_to_matching_matrix(GT_matching_mask)

    # compute all the possible masks
    matching_matrix = mutual_nearest_neighbors_from_score_matrix(
        score_matrix_des, min_score=min_score, ratio_test=ratio_test
    )  # B,n_kpts0,n_kpts1
    matching_matrix_agg = compute_correct_wrong_mismatched_inexistent_unsure_matches(
        matching_matrix, matching_matrix_GT_with_bins
    )

    # compute the sum
    n_matches_gt = GT_matching_mask.sum() / B
    n_matches_proposed = matching_matrix_agg.proposed.sum() / B
    n_matches_correct = matching_matrix_agg.correct.sum() / B
    n_matches_wrong = matching_matrix_agg.wrong.sum() / B
    n_matches_mismatched = matching_matrix_agg.mismatched.sum() / B
    n_matches_inexistent = matching_matrix_agg.inexistent.sum() / B
    n_matches_unsure = matching_matrix_agg.unsure.sum() / B

    # index the scores
    scores_GT_matches = score_matrix_des[GT_matching_mask]  # n_matches_gt_fullbatch
    scores_all = score_matrix_des.view(-1)
    scores_negative_all = score_matrix_des[~GT_matching_mask]
    scores_matches_proposed = score_matrix_des[matching_matrix_agg.proposed]
    scores_matches_correct = score_matrix_des[matching_matrix_agg.correct]
    scores_matches_wrong = score_matrix_des[matching_matrix_agg.wrong]
    scores_matches_mismatched = score_matrix_des[matching_matrix_agg.mismatched]
    scores_matches_inexistent = score_matrix_des[matching_matrix_agg.inexistent]

    # remove nan values
    scores_all = scores_all[~torch.isnan(scores_all)]
    scores_negative_all = scores_negative_all[~torch.isnan(scores_negative_all)]

    if scores_GT_matches.isnan().any():
        print("WARNING: nan values in scores_GT_matches, this should never happen")
    if scores_matches_proposed.isnan().any():
        print(
            "WARNING: nan values in scores_matches_proposed, this should never happen"
        )
    if scores_matches_correct.isnan().any():
        print("WARNING: nan values in scores_matches_correct, this should never happen")
    if scores_matches_wrong.isnan().any():
        print("WARNING: nan values in scores_matches_wrong, this should never happen")
    if scores_matches_mismatched.isnan().any():
        print(
            "WARNING: nan values in scores_matches_mismatched, this should never happen"
        )
    if scores_matches_inexistent.isnan().any():
        print(
            "WARNING: nan values in scores_matches_inexistent, this should never happen"
        )

    # compute stats
    matches_precision = n_matches_correct / n_matches_proposed
    matches_recall = n_matches_correct / n_matches_gt

    # compute the margins
    score_matrix_des_with_inf = score_matrix_des.clone()
    score_matrix_des_with_inf[score_matrix_des_with_inf.isnan()] = float("-inf")

    best_two_scores0 = torch.topk(score_matrix_des_with_inf, 2, dim=-1)[0]
    best_two_scores1 = torch.topk(score_matrix_des_with_inf, 2, dim=-2)[0]
    best_scores0, second_best_scores0 = (
        best_two_scores0[:, :, 0],
        best_two_scores0[:, :, 1],
    )  # (B,n_kpts0) (B,n_kpts0)
    best_scores1, second_best_scores1 = (
        best_two_scores1[:, 0, :],
        best_two_scores1[:, 1, :],
    )  # (B,n_kpts1) (B,n_kpts1)

    # margin for all the proposed matches
    margin_proposed, ratio_proposed = get_margin_and_ratio_from_scores_and_mnn_matrix(
        matching_matrix_agg.proposed,
        best_scores0,
        second_best_scores0,
        second_best_scores1,
    )
    # correct matches margin
    margin_correct, ratio_correct = get_margin_and_ratio_from_scores_and_mnn_matrix(
        matching_matrix_agg.correct,
        best_scores0,
        second_best_scores0,
        second_best_scores1,
    )
    # wrong matches margin
    margin_wrong, ratio_wrong = get_margin_and_ratio_from_scores_and_mnn_matrix(
        matching_matrix_agg.wrong,
        best_scores0,
        second_best_scores0,
        second_best_scores1,
    )
    # mismatched matches margin
    margin_mismatched, ratio_mismatched = (
        get_margin_and_ratio_from_scores_and_mnn_matrix(
            matching_matrix_agg.mismatched,
            best_scores0,
            second_best_scores0,
            second_best_scores1,
        )
    )
    # inexistent matches margin
    margin_inexistent, ratio_inexistent = (
        get_margin_and_ratio_from_scores_and_mnn_matrix(
            matching_matrix_agg.inexistent,
            best_scores0,
            second_best_scores0,
            second_best_scores1,
        )
    )

    # find out how many possible mismatched have been shielded by a correct match
    # we do this counting how many column have the max score that correspond to a column where there is a correct match
    matches_correct_idx = (
        matching_matrix_agg.correct.nonzero()
    )  # (n_matches_correct, 2)
    # we first create a mask with a one in the position where the score is the max for that row
    row_max_mask = (
        score_matrix_des_with_inf
        == score_matrix_des_with_inf.max(dim=-1, keepdim=True)[0]
    ) * score_matrix_des_with_inf.isfinite()  # (B,n_kpts0,n_kpts1)
    # we then index only the columns where there was a correct match, and sum over those columns
    # (subtracting always one as we do not want to count the correct match)
    masked_columns = row_max_mask[
        matches_correct_idx[:, 0], :, matches_correct_idx[:, 2]
    ]  # n_masked_columns, n_kpts0
    n_masked_by_columns = masked_columns.sum() - masked_columns.shape[0]
    # do the same by columns
    column_max_mask = (
        score_matrix_des_with_inf
        == score_matrix_des_with_inf.max(dim=-2, keepdim=True)[0]
    ) * score_matrix_des_with_inf.isfinite()  # (B,n_kpts0,n_kpts1)
    masked_rows = column_max_mask[
        matches_correct_idx[:, 0], matches_correct_idx[:, 1], :
    ]  # n_masked_rows, n_kpts1
    n_masked_by_rows = masked_rows.sum() - masked_rows.shape[0]
    n_masked = n_masked_by_columns + n_masked_by_rows

    # # reorder the descriptors of batch0 to generate a nice score matrix to log
    # idx0, idx1 = torch.where(GT_matching_mask[0])  # (n_matches_gt_batch0), (n_matches_gt_batch0)
    # mask0_not_matched = torch.ones(n_kpts0, dtype=torch.bool, device=DEVICE)
    # mask1_not_matched = torch.ones(n_kpts1, dtype=torch.bool, device=DEVICE)
    # mask0_not_matched[idx0] = False
    # mask1_not_matched[idx1] = False
    # des0_batch0_ordered = torch.cat((des0[0][idx0], des0[0][mask0_not_matched]))  # n_kpts0,des_dim
    # des1_batch0_ordered = torch.cat((des1[0][idx1], des1[0][mask1_not_matched]))  # n_kpts1,des_dim
    # score_matrix_batch0 = des0_batch0_ordered @ des1_batch0_ordered.T  # n_kpts0,n_kpts1

    numeric_stats = {
        "n_matches_proposed": int(n_matches_proposed.item()),
        "n_matches_correct": int(n_matches_correct.item()),
        "n_matches_wrong": int(n_matches_wrong.item()),
        "n_matches_mismatched": int(n_matches_mismatched.item()),
        "n_matches_inexistent": int(n_matches_inexistent.item()),
        "n_matches_unsure": int(n_matches_unsure.item()),
        "n_matches_masked": int(n_masked.item()),
        "matches_precision": matches_precision.item(),
        "matches_recall": matches_recall.item(),
        "avg_positive_score": scores_GT_matches.mean().item(),
        "avg_negative_score_all": scores_negative_all.mean().item(),
        "avg_score_matches_proposed": scores_matches_proposed.mean().item(),
        "avg_score_matches_correct": scores_matches_correct.mean().item(),
        "avg_score_matches_wrong": scores_matches_wrong.mean().item(),
        "avg_score_matches_mismatched": scores_matches_mismatched.mean().item(),
        "avg_score_matches_inexistent": scores_matches_inexistent.mean().item(),
        "avg_margin_proposed": margin_proposed.mean().item(),
        "avg_margin_correct": margin_correct.mean().item(),
        "avg_margin_wrong": margin_wrong.mean().item(),
        "avg_margin_mismatched": margin_mismatched.mean().item(),
        "avg_margin_inexistent": margin_inexistent.mean().item(),
        "avg_ratio_proposed": ratio_proposed.mean().item(),
        "avg_ratio_correct": ratio_correct.mean().item(),
        "avg_ratio_wrong": ratio_wrong.mean().item(),
        "avg_ratio_mismatched": ratio_mismatched.mean().item(),
        "avg_ratio_inexistent": ratio_inexistent.mean().item(),
    }

    tensor_stats = {
        "score_all": scores_all,
        "score_GT_matches": scores_GT_matches,
        "score_negative_all": scores_negative_all,
        "score_matches_proposed": scores_matches_proposed,
        "score_matches_correct": scores_matches_correct,
        "score_matches_wrong": scores_matches_wrong,
        "score_matches_mismatched": scores_matches_mismatched,
        "score_matches_inexistent": scores_matches_inexistent,
        "margin_proposed": margin_proposed,
        "margin_correct": margin_correct,
        "margin_wrong": margin_wrong,
        "margin_mismatched": margin_mismatched,
        "margin_inexistent": margin_inexistent,
        "ratio_proposed": ratio_proposed,
        "ratio_correct": ratio_correct,
        "ratio_wrong": ratio_wrong,
        "ratio_mismatched": ratio_mismatched,
        "ratio_inexistent": ratio_inexistent,
    }

    return numeric_stats, tensor_stats
