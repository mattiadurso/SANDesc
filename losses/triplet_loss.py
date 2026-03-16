import torch
from torch import nn
from torch import Tensor


class TripletLoss(nn.Module):
    """Computes the triplet loss with the provided sampled descriptors:"""

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
        """Initializes the TripletLoss module

        Args:
            margin: the margin for the triplet loss
            ratio: the ratio for the triplet loss (works similar to the margin, but it's a relative threshold)
            random_negative_ratio: if > 0, it will sample random negatives with this probability instead of the hardest
            random_negative_ratio_decay: the decay of the random negative ratio
            quadratic: whether to square the loss as in SOSNet paper
            weight_by_keypoints_score: whether to weight the loss by the keypoint score
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
        _kpts_scores: Tensor = None,
    ) -> Tensor:
        """Compute the triplet loss

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
            # chosen_triplets_margin = scores_pos - scores_neg > self.margin
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

        _loss = (_loss**2).mean() if self.quadratic else _loss.mean()

        return _loss

    def __repr__(self) -> str:
        return f"{self.name}\n   margin: {self.m}"

    def get_hardest_triplets(
        self, des0: Tensor, des1: Tensor, matches_matrix_GT_with_bins: Tensor
    ) -> Tensor:
        """Find the hardest triplets for the given descriptors and the GT matches

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
            with torch.no_grad():
                matches_from_kpts = (
                    matches_matrix_GT_with_bins[b, :-1, :-1]
                ).nonzero()  # n_matches_kpts,2
                n_matches_kpts = matches_from_kpts.shape[0]
                if n_matches_kpts == 0:
                    if self.verbose:
                        print("no matches GT for this batch index")
                    continue

                # Get anchors
                idx_anchor0 = matches_from_kpts[:, 0]  # n_matches_kpts
                idx_anchor1 = matches_from_kpts[:, 1]  # n_matches_kpts

                # Get positives
                idx_pos0 = idx_anchor1.clone()  # n_matches_kpts
                idx_pos1 = idx_anchor0.clone()  # n_matches_kpts

                if (
                    matches_matrix_GT_with_bins[b, :-1, :-1].sum(-1).max() > 1
                    or matches_matrix_GT_with_bins[b, :-1, :-1].sum(-2).max() > 1
                ):
                    # we have more than one match for keypoints (because we run the detector multiscale), we
                    # want to take the best possible match for each row and column (without actually requiring to
                    # be mutual best match)
                    scores = torch.zeros((n0, n1), device=device)
                    scores[idx_anchor0, idx_anchor1] = (
                        des0[b][idx_anchor0] * des1[b][idx_pos0]
                    ).sum(
                        -1
                    )  # n_matches_kpts,n_matches_kpts
                    # keep only the scores that are max by row or by column
                    mask_rows = scores == scores.max(-1, keepdim=True)[0]
                    mask_columns = scores == scores.max(-2, keepdim=True)[0]
                    # scores_best = scores * (mask_rows + mask_columns)
                    scores_best = scores * (mask_rows * mask_columns)

                    # re-find the GT matches from this new score matrix
                    best_matches_from_kpts = scores_best.nonzero()  # n_matches_kpts,2
                    idx_anchor0 = best_matches_from_kpts[:, 0]  # n_matches_kpts
                    idx_anchor1 = best_matches_from_kpts[:, 1]  # n_matches_kpts
                    idx_pos0 = idx_anchor1.clone()  # n_matches_kpts
                    idx_pos1 = idx_anchor0.clone()  # n_matches_kpts
                    n_matches_kpts = best_matches_from_kpts.shape[0]
                    print(
                        f"delta n matches: {matches_from_kpts.shape[0] - best_matches_from_kpts.shape[0]}"
                    )

            anchor0 = des0[b][idx_anchor0]  # n_matches_kpts,des_dim
            anchor1 = des1[b][idx_anchor1]  # n_matches_kpts,des_dim
            pos0 = des1[b][idx_pos0]  # n_matches_kpts,des_dim
            pos1 = des0[b][idx_pos1]  # n_matches_kpts,des_dim

            with torch.no_grad():
                # we put as negative all the descriptors from img1 which are not the positive
                # we do this for the single batch
                negatives_all0 = des1[b].detach()  # n_kpts,des_dim
                negatives_all1 = des0[b].detach()  # n_kpts,des_dim
                # compute the anchor-negative scores
                scores_neg0 = anchor0 @ negatives_all0.T  # n_matches_kpts,n_kpts
                scores_neg1 = anchor1 @ negatives_all1.T  # n_matches_kpts,n_kpts
                # remove the actual positives from the scores (this works also with multiple positives)
                scores_neg0[matches_matrix_GT_with_bins[b, :-1, :-1][idx_anchor0]] = (
                    float("-inf")
                )
                scores_neg1[matches_matrix_GT_with_bins[b, :-1, :-1].T[idx_anchor1]] = (
                    float("-inf")
                )
                # remove the invalid descriptors
                scores_neg0[scores_neg0.isnan()] = float("-inf")
                scores_neg1[scores_neg1.isnan()] = float("-inf")
                # find the (since removed the positives, second) hardest
                score0_hardest, idx_hardest0 = scores_neg0.max(-1)  # n_matches_kpts
                score1_hardest, idx_hardest1 = scores_neg1.max(-1)  # n_matches_kpts

                if score0_hardest.isnan().any() or score1_hardest.isnan().any():
                    print(
                        "WARNING, one of the score_hardest is nan, this should never happen"
                    )
                    continue

                # random_negative_ratio starts from 1 and then decays. This means at the beginning the model has no
                # negatives and pushes all the descriptors to the same point in the embedding space. Then,
                # as the random_negative_ratio decays, the model starts to use the hardest negatives.
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
                print(
                    "WARNING, one of the idx_pos is equal to idx_hardest, this should never happen"
                )
                if (idx_pos0 == idx_hardest0).all():
                    print(idx_pos0)
                    print(idx_hardest0)
                if (idx_pos1 == idx_hardest1).all():
                    print(idx_pos1)
                    print(idx_hardest1)
                continue

            des1_valid = des1[b][~torch.isnan(des1[b]).any(dim=1)]  # n1,des_dim
            des0_valid = des0[b][~torch.isnan(des0[b]).any(dim=1)]  # n1,des_dim
            if des1_valid.shape[0] > 0:
                random_idx = torch.randint(
                    0, des1_valid.shape[0], (n_matches_kpts,), device=device
                )
                neg0 = torch.where(
                    mask0, des1_valid[random_idx], des1[b][idx_hardest0]
                )  # n_matches_kpts,des_dim
            else:
                print("WARNING, no valid descriptors in image 1")
                neg0 = torch.zeros(
                    n_matches_kpts, des1.shape[-1], device=des1.device
                ) * float(
                    "nan"
                )  # n_matches_kpts,des_dim
            if des0_valid.shape[0] > 0:
                random_idx = torch.randint(
                    0, des0_valid.shape[0], (n_matches_kpts,), device=device
                )
                neg1 = torch.where(
                    mask1, des0_valid[random_idx], des0[b][idx_hardest1]
                )  # n_matches_kpts,des_dim
            else:
                print("WARNING, no valid descriptors in image 1")
                neg1 = torch.zeros(
                    n_matches_kpts, des0.shape[-1], device=des0.device
                ) * float(
                    "nan"
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
    def compute_triplets_stats(self, triplets: Tensor):
        """computes some useful statistics about the triplets
        Args:
            triplets: the triplets tensor
                n_triplets,3,des_dim

        Returns:
            stats: a dictionary with the statistics
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
