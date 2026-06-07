"""Triplet loss with hardest-negative mining for descriptor training."""

import logging

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    """Computes the triplet loss with the provided sampled descriptors."""

    def __init__(
        self,
        margin: float,
        ratio: float = 1.0,
        random_negative_ratio: float = 0.0,
        random_negative_ratio_decay: float = 0.0,
        quadratic: bool = False,
        verbose: bool = False,
        weight_by_keypoints_score: bool = False,
    ) -> None:
        """Initializes the TripletLoss module.

        Args:
            margin: the margin for the triplet loss.
            ratio: the ratio for the triplet loss (works like the margin, but
                as a relative threshold).
            random_negative_ratio: if > 0, sample random negatives with this
                probability instead of the hardest.
            random_negative_ratio_decay: the decay of the random negative ratio.
            quadratic: whether to square the loss as in the SOSNet paper.
            verbose: whether to print diagnostic messages.
            weight_by_keypoints_score: whether to weight the loss by the
                keypoint score.
        """
        super().__init__()
        self.name = "TripletLoss"
        self.margin = margin
        self.ratio = ratio
        self.random_negative_ratio = random_negative_ratio
        self.random_negative_ratio_decay = random_negative_ratio_decay
        self.quadratic = quadratic
        self.verbose = verbose
        self.weight_by_keypoint_score = weight_by_keypoints_score

    def forward(
        self,
        des_anchor: Tensor,
        des_pos: Tensor,
        des_neg: Tensor,
        _kpts_scores: Tensor | None = None,
    ) -> Tensor:
        """Compute the triplet loss.

        Args:
            des_anchor: the anchor descriptors
                na,des_dim
            des_pos: the positive descriptors
                na,des_dim
            des_neg: the negative descriptors
                na,des_dim
            _kpts_scores: the keypoint scores

        Returns:
            x: the triplet loss

        """
        scores_pos = (des_anchor * des_pos).sum(-1)  # na
        scores_neg = (des_anchor * des_neg).sum(-1)  # na

        if self.margin > 0.0:
            chosen_triplets_margin = scores_pos - self.margin < scores_neg
        else:
            chosen_triplets_margin = torch.ones_like(scores_pos, dtype=torch.bool)

        if self.ratio < 1.0:
            chosen_triplets_ratio = scores_neg / scores_pos > self.ratio
        else:
            chosen_triplets_ratio = torch.ones_like(scores_pos, dtype=torch.bool)

        chosen_triplets = chosen_triplets_margin * chosen_triplets_ratio

        _loss = scores_neg[chosen_triplets] - scores_pos[chosen_triplets]  # n_triplets

        if _kpts_scores is not None and self.weight_by_keypoint_score:
            _loss *= _kpts_scores

        return (_loss**2).mean() if self.quadratic else _loss.mean()

    def __repr__(self) -> str:
        """Return a short representation with the loss name and margin."""
        return f"{self.name}\n   margin: {self.margin}"

    @staticmethod
    def _resolve_multiscale_anchors(
        des0: Tensor,
        des1: Tensor,
        gt_mask: Tensor,
        idx_anchor0: Tensor,
        idx_anchor1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pick the best single match per row/column for multiscale GT.

        When the detector runs multiscale there can be several GT matches per
        keypoint; keep, for each row and column, the highest-scoring pair
        (without requiring it to be a mutual best match). Returns the new
        ``(idx_anchor0, idx_anchor1)``.
        """
        n0, n1 = gt_mask.shape
        device = des0.device
        scores = torch.zeros((n0, n1), device=device)
        scores[idx_anchor0, idx_anchor1] = (des0[idx_anchor0] * des1[idx_anchor1]).sum(
            -1
        )
        # keep only the scores that are max by row or by column
        mask_rows = scores == scores.max(-1, keepdim=True)[0]
        mask_columns = scores == scores.max(-2, keepdim=True)[0]
        scores_best = scores * (mask_rows * mask_columns)
        best_matches_from_kpts = scores_best.nonzero()  # n_matches_kpts,2
        return best_matches_from_kpts[:, 0], best_matches_from_kpts[:, 1]

    @staticmethod
    def _hardest_negative_indices(
        anchor: Tensor,
        negatives_all: Tensor,
        positives_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Find the hardest (highest-scoring) non-positive negative per anchor.

        Returns ``(score_hardest, idx_hardest)``; positives and invalid (nan)
        scores are masked out before taking the max.
        """
        scores_neg = anchor @ negatives_all.T  # n_matches_kpts,n_kpts
        # remove the actual positives (works also with multiple positives)
        scores_neg[positives_mask] = float("-inf")
        # remove the invalid descriptors
        scores_neg[scores_neg.isnan()] = float("-inf")
        return scores_neg.max(-1)  # (score, idx)

    @staticmethod
    def _sample_negative(
        des_negatives: Tensor,
        idx_hardest: Tensor,
        random_mask: Tensor,
        n_matches_kpts: int,
        device: torch.device,
    ) -> Tensor:
        """Pick the negative descriptor: random valid one where mask is True.

        Falls back to all-NaN when no valid negative descriptor exists.
        """
        des_valid = des_negatives[~torch.isnan(des_negatives).any(dim=1)]
        if des_valid.shape[0] > 0:
            random_idx = torch.randint(
                0, des_valid.shape[0], (n_matches_kpts,), device=device
            )
            return torch.where(
                random_mask, des_valid[random_idx], des_negatives[idx_hardest]
            )
        log.warning("no valid descriptors in image 1")
        return torch.full(
            (n_matches_kpts, des_negatives.shape[-1]),
            float("nan"),
            device=des_negatives.device,
        )

    def get_hardest_triplets(
        self, des0: Tensor, des1: Tensor, matches_matrix_GT_with_bins: Tensor
    ) -> Tensor:
        """Find the hardest triplets for the given descriptors and the GT matches.

        Args:
            des0: the descriptors from the first image
                B,n0,des_dim
            des1: the descriptors from the second image
                B,n1,des_dim
            matches_matrix_GT_with_bins: the GT matches matrix
                B,n0+1,n1+1

        Returns:
            triplets: the triplets tensor
                n_triplets,3,des_dim

        """
        B, n0, des_dim = des0.shape
        _, n1, _ = des1.shape
        device = des0.device
        triplets = torch.tensor([], device=device)  # n_triplets,3,des_dim
        for b in range(B):
            gt_mask = matches_matrix_GT_with_bins[b, :-1, :-1]  # n0,n1 bool
            with torch.no_grad():
                matches_from_kpts = gt_mask.nonzero()  # n_matches_kpts,2
                n_matches_kpts = matches_from_kpts.shape[0]
                if n_matches_kpts == 0:
                    if self.verbose:
                        log.debug("no matches GT for this batch index")
                    continue

                # anchors and positives (positive of img0 anchor is its img1 match)
                idx_anchor0 = matches_from_kpts[:, 0]  # n_matches_kpts
                idx_anchor1 = matches_from_kpts[:, 1]  # n_matches_kpts

                if gt_mask.sum(-1).max() > 1 or gt_mask.sum(-2).max() > 1:
                    # multiscale: more than one match per keypoint
                    idx_anchor0, idx_anchor1 = self._resolve_multiscale_anchors(
                        des0[b], des1[b], gt_mask, idx_anchor0, idx_anchor1
                    )
                    delta_n = n_matches_kpts - idx_anchor0.shape[0]
                    n_matches_kpts = idx_anchor0.shape[0]
                    if self.verbose:
                        log.debug("delta n matches: %d", delta_n)

                idx_pos0 = idx_anchor1.clone()  # n_matches_kpts
                idx_pos1 = idx_anchor0.clone()  # n_matches_kpts

            anchor0 = des0[b][idx_anchor0]  # n_matches_kpts,des_dim
            anchor1 = des1[b][idx_anchor1]  # n_matches_kpts,des_dim
            pos0 = des1[b][idx_pos0]  # n_matches_kpts,des_dim
            pos1 = des0[b][idx_pos1]  # n_matches_kpts,des_dim

            with torch.no_grad():
                # all descriptors from the other image (except the positive) are
                # candidate negatives; find the hardest for each anchor
                score0_hardest, idx_hardest0 = self._hardest_negative_indices(
                    anchor0, des1[b].detach(), gt_mask[idx_anchor0]
                )
                score1_hardest, idx_hardest1 = self._hardest_negative_indices(
                    anchor1, des0[b].detach(), gt_mask.T[idx_anchor1]
                )

                if score0_hardest.isnan().any() or score1_hardest.isnan().any():
                    log.warning(
                        "one of the score_hardest is nan, this should never happen"
                    )
                    continue

                # random_negative_ratio starts from 1 and then decays. This
                # means at the beginning the model has no negatives and pushes
                # all the descriptors to the same point in the embedding space.
                # Then, as the random_negative_ratio decays, the model starts
                # to use the hardest negatives.
                mask0 = (
                    torch.rand(n_matches_kpts, device=des0.device)
                    < self.random_negative_ratio
                )  # n_matches
                mask1 = (
                    torch.rand(n_matches_kpts, device=des1.device)
                    < self.random_negative_ratio
                )  # n_matches
                mask0 = mask0[:, None].repeat(
                    1, des0[b].shape[1]
                )  # n_matches_kpts,des_dim
                mask1 = mask1[:, None].repeat(
                    1, des1[b].shape[1]
                )  # n_matches_kpts,des_dim

            if (idx_pos0 == idx_hardest0).any() or (idx_pos1 == idx_hardest1).any():
                log.warning(
                    "one of the idx_pos is equal to idx_hardest, this should "
                    "never happen"
                )
                if self.verbose and (idx_pos0 == idx_hardest0).all():
                    log.debug("idx_pos0=%s idx_hardest0=%s", idx_pos0, idx_hardest0)
                if self.verbose and (idx_pos1 == idx_hardest1).all():
                    log.debug("idx_pos1=%s idx_hardest1=%s", idx_pos1, idx_hardest1)
                continue

            neg0 = self._sample_negative(
                des1[b], idx_hardest0, mask0, n_matches_kpts, device
            )  # n_matches_kpts,des_dim
            neg1 = self._sample_negative(
                des0[b], idx_hardest1, mask1, n_matches_kpts, device
            )  # n_matches_kpts,des_dim

            # stack and concatenate
            triplets_b0 = torch.stack(
                [anchor0, pos0, neg0], dim=1
            )  # n_matches_kpts,3,des_dim
            triplets_b1 = torch.stack(
                [anchor1, pos1, neg1], dim=1
            )  # n_matches_kpts,3,des_dim

            triplets = torch.cat(
                [triplets, triplets_b0, triplets_b1], dim=0
            )  # n_cumulative_matches,3,des_dim

        # update the random negative ratio
        self.random_negative_ratio = (
            self.random_negative_ratio * self.random_negative_ratio_decay
        )
        return triplets

    @torch.no_grad()
    def compute_triplets_stats(self, triplets: Tensor) -> dict[str, float]:
        """Compute some useful statistics about the triplets.

        Args:
            triplets: the triplets tensor
                n_triplets,3,des_dim.

        Returns:
            A dictionary with the statistics.
        """
        positive_score = (triplets[:, 0] * triplets[:, 1]).sum(-1)  # n_triplets
        negative_score_triplets = (triplets[:, 0] * triplets[:, 2]).sum(
            -1
        )  # n_triplets
        avg_negative_score_triplets = negative_score_triplets.mean()
        avg_margin_triplet = (positive_score - negative_score_triplets).mean()
        avg_ratio_triplet = (negative_score_triplets / positive_score).mean()
        return {
            "avg_negative_score_triplet": avg_negative_score_triplets.item(),
            "avg_margin_triplet": avg_margin_triplet.item(),
            "avg_ratio_triplet": avg_ratio_triplet.item(),
            "random_negative_ratio": self.random_negative_ratio,
        }
