import time
import wandb
import hydra
import torch
import logging
import datetime
import torch.nn.functional as F

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Training utils imports
from datasets.dataset_loaders import (
    transform_from_normalized_rgb_to_grayscale,
)

from utils.descriptor_stats import compute_stats
from utils.descriptors_utils import extract_keypoints, evaluate
from utils.norm import compute_grad_norm
from utils.utils_logging import log_match_plot
from utils.seed_control import seed_management
from utils.utils_network import save_checkpoint
from utils.helpers import (
    sanitize_config_for_omegaconf,
    load_checkpoint_if_needed,
    setup_dataloaders,
    setup_model_and_optimizer,
    setup_loss_and_scaler,
    setup_wrappers,
    setup_logging,
    set_deterministic_behavior,
    setup_paths,
)

setup_paths()

# Get logger
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    log.info("Configuration:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Setup
    set_deterministic_behavior()
    seed_management("reset")
    checkpoint = load_checkpoint_if_needed(cfg.resume_from)

    # Use checkpoint config if resuming
    if checkpoint is not None:
        try:
            # Sanitize the checkpoint config to handle torch.dtype and other non-primitive types
            sanitized_checkpoint_config = sanitize_config_for_omegaconf(
                checkpoint["config"]
            )
            cfg = OmegaConf.create(sanitized_checkpoint_config)

            # Only preserve device and resume_from from original config
            original_resume_from = cfg.get("resume_from")
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.resume_from = original_resume_from

            log.info("Successfully loaded checkpoint config")

        except Exception as e:
            log.warning(f"Could not load checkpoint config: {e}")
            log.info("Continuing with current config...")

    log.info("Final config:")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Setup components
    (
        train_dataloader,
        valid_dataloader,
        valid_dataloader_hard,
        compute_GT_matching_matrix_fn,
        compute_GT_matching_matrix_fn_validation,
    ) = setup_dataloaders(cfg)

    # Setup loss BEFORE model init so we can pass it to checkpoint loading
    triplet_loss_fn, scaler, amp_device_type, amp_dtype = setup_loss_and_scaler(cfg)

    network, optimizer, lr_scheduler, iteration = setup_model_and_optimizer(
        cfg, checkpoint
    )

    # Load triplet loss state from checkpoint
    if checkpoint is not None and "triplet_loss_state" in checkpoint:
        triplet_loss_fn.random_negative_ratio = checkpoint["triplet_loss_state"].get(
            "random_negative_ratio", triplet_loss_fn.random_negative_ratio
        )
        triplet_loss_fn.random_negative_ratio_decay = checkpoint[
            "triplet_loss_state"
        ].get(
            "random_negative_ratio_decay", triplet_loss_fn.random_negative_ratio_decay
        )
        log.info(
            f"Loaded triplet loss state: random_negative_ratio={triplet_loss_fn.random_negative_ratio}"
        )

    train_wrapper, valid_wrapper = setup_wrappers(cfg)
    setup_logging(cfg, checkpoint)

    # Model summary
    log.info("Model summary:")
    try:
        from torchinfo import summary

        summary(network)
    except Exception as e:
        log.warning(f"Could not summarize model: {e}")
        log.info("Continuing without model summary...")

    if cfg.use_wandb and wandb.run is not None:
        wandb.watch(network, log="all", log_graph=True, log_freq=100)

    # Restore random state if resuming
    if checkpoint is not None:
        try:
            seed_management("restore", checkpoint["random_state"])
            log.info(f"Resumed from iteration {checkpoint['iteration']:,}")
        except:
            log.warning("Random states not found in the checkpoint")
        seed_management("reset")

    # Training variables
    start_time = time.time()
    total_images = checkpoint.get("total_images", 0) if checkpoint else 0

    # Training loop
    try:
        for _ in range(10):
            log.info(">>> Training...")

            for data in train_dataloader:
                # Evaluation
                if (
                    cfg.training.eval
                    and iteration % cfg.training.evaluate_every_n_iterations == 0
                    # and iteration > 0
                ):

                    log.info(">>> Evaluating on IMB validation set...")
                    valid_wrapper.multiscale = cfg.training.validation_multiscale

                    # Normal validation
                    evaluate(
                        valid_wrapper,
                        network,
                        valid_dataloader,
                        compute_GT_matching_matrix_fn_validation,
                        current_interation=iteration,
                        n_max_keypoints=2048,
                        n_iterations=cfg.training.n_images_stats,
                        n_plots=cfg.training.n_images_to_log,
                        device=cfg.device,
                        tag="valid/",
                        use_wrapper_descriptor=False,
                        compute_pose_stats=True,
                        ratio_test=cfg.training.ratio_test,
                        max_epipolar_error=cfg.training.max_epipolar_error,
                        return_stats=False,
                    )

                    # Hard validation
                    evaluate(
                        valid_wrapper,
                        network,
                        valid_dataloader_hard,
                        compute_GT_matching_matrix_fn_validation,
                        current_interation=iteration,
                        n_max_keypoints=2048,
                        n_iterations=cfg.training.n_images_stats,
                        n_plots=cfg.training.n_images_to_log,
                        device=cfg.device,
                        tag="valid_hard/",
                        use_wrapper_descriptor=False,
                        compute_pose_stats=True,
                        ratio_test=cfg.training.ratio_test,
                        max_epipolar_error=cfg.training.max_epipolar_error,
                        return_stats=False,
                    )

                    # # MD1500 validation
                    # evaluate(
                    #     ...
                    # )

                    valid_wrapper.multiscale = False

                # Training step
                B, C, H, W = data["img0"].shape
                mask = torch.ones(B, dtype=torch.bool)

                # Extract keypoints
                if cfg.training.train_wrapper == "random":
                    kpts0, kpts1 = train_wrapper.extract(
                        data, cfg.training.max_n_keypoints
                    )
                    mask_0kpts = kpts0.sum(dim=-1).sum(dim=-1) > 0
                    mask_1kpts = kpts1.sum(dim=-1).sum(dim=-1) > 0
                    mask = mask_0kpts & mask_1kpts
                    kpts0 = kpts0[mask].to(cfg.device)
                    kpts1 = kpts1[mask].to(cfg.device)

                # Move data to device
                for key in data.keys():
                    if key in data:
                        data[key] = data[key][mask].to(cfg.device)

                img0, img1 = data["img0"], data["img1"]

                # Extract keypoints for non-random wrappers
                if cfg.training.train_wrapper != "random":
                    kpts0, kpts1 = extract_keypoints(
                        img0, img1, train_wrapper, cfg.training.max_n_keypoints
                    )[:2]

                # Forward pass
                with torch.autocast(
                    device_type=amp_device_type,
                    dtype=amp_dtype,
                    enabled=cfg.training.use_amp,
                ):
                    des_vol0 = network(img0)
                    des_vol1 = network(img1)

                # Sample descriptors | with nearest I avoid to renormalize descriptors
                des0 = train_wrapper.grid_sample_nan(kpts0, des_vol0, mode="nearest")[
                    0
                ].permute(0, 2, 1)
                des1 = train_wrapper.grid_sample_nan(kpts1, des_vol1, mode="nearest")[
                    0
                ].permute(0, 2, 1)

                # Compute GT matches
                data["kpts0"] = kpts0
                data["kpts1"] = kpts1
                matching_matrix_GT_with_bins = compute_GT_matching_matrix_fn(data)

                # Loss computation
                triplets = triplet_loss_fn.get_hardest_triplets(
                    des0, des1, matching_matrix_GT_with_bins
                )

                if triplets.shape[0] == 0:
                    log.warning("No triplets found, skipping iteration")
                    continue

                with torch.autocast(
                    device_type=amp_device_type,
                    dtype=amp_dtype,
                    enabled=cfg.training.use_amp,
                ):
                    loss = triplet_loss_fn(
                        triplets[:, 0], triplets[:, 1], triplets[:, 2]
                    )

                # Backward pass
                scaler.scale(loss).backward()
                grad_norm = compute_grad_norm(network)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update learning rate
                old_lr = optimizer.param_groups[0]["lr"]
                lr_scheduler.update_lr(optimizer, iteration)

                # Logging
                with torch.no_grad():
                    score_matrix = des0 @ des1.permute(0, 2, 1)
                    stats = compute_stats(score_matrix, matching_matrix_GT_with_bins)[0]
                    stats.update(triplet_loss_fn.compute_triplets_stats(triplets))
                    stats["loss_triplet"] = loss.item()
                    stats["grad_norm"] = grad_norm

                log_data = {
                    "loss": loss.item(),
                    "n_training_matches": matching_matrix_GT_with_bins[
                        :, :-1, :-1
                    ].sum()
                    / B,
                    "n_training_bin_matches": (
                        matching_matrix_GT_with_bins[:, :, -1].sum()
                        + matching_matrix_GT_with_bins[:, -1, :].sum()
                    )
                    / B,
                    "n_kpts": 0.5
                    * (
                        (~kpts0[:, :, 0].isnan()).sum()
                        + (~kpts1[:, :, 0].isnan()).sum()
                    )
                    / B,
                    "lr": old_lr,
                    **stats,
                }

                if cfg.use_wandb and wandb.run is not None:
                    wandb.log(log_data, step=iteration)

                # Plotting
                if iteration % 100 == 0:
                    if img0.shape[1] == 3:
                        img0 = transform_from_normalized_rgb_to_grayscale(img0)
                        img1 = transform_from_normalized_rgb_to_grayscale(img1)

                    log_match_plot(
                        img0[0:1],
                        img1[0:1],
                        kpts0[0:1],
                        kpts1[0:1],
                        score_matrix[0:1],
                        matching_matrix_GT_with_bins[0:1],
                        batch_idx=0,
                        iteration=iteration,
                        tag="train_",
                        caption=f"img-{iteration}\nn_matches_GT: {matching_matrix_GT_with_bins[0, :-1, :-1].sum()}",
                    )

                # Checkpointing
                if (
                    iteration + 1
                ) % cfg.training.save_every_n_iterations == 0 and iteration > 0:
                    save_dir = (
                        Path(wandb.run.dir)
                        if (cfg.use_wandb and wandb.run)
                        else Path(cfg.save_path)
                    )

                    # Use original config structure (it should already have triplet_loss)
                    save_config = OmegaConf.to_container(cfg, resolve=True)

                    # Only sanitize the dtype if needed
                    if isinstance(save_config["training"]["amp_dtype"], torch.dtype):
                        save_config["training"]["amp_dtype"] = str(
                            save_config["training"]["amp_dtype"]
                        ).split(".")[-1]

                    save_checkpoint(
                        save_config,
                        network,
                        optimizer,
                        iteration + 1,
                        save_dir,
                        triplet_loss=triplet_loss_fn,
                        random_states=seed_management("store"),
                        save_all=(iteration + 1)
                        % cfg.training.evaluate_every_n_iterations
                        == 0,
                        total_images=total_images,
                    )

                # Progress logging
                total_images += img0.shape[0]
                elapsed = str(
                    datetime.timedelta(seconds=time.time() - start_time)
                ).split(".")[0]
                log.info(
                    f"{iteration+1:,}/{cfg.training.max_iterations:,}, elapsed: {elapsed} | "
                    f"loss: {loss.item():.4f}, lr: {old_lr:.4f}, pairs: {mask.sum().item()}/{B}, "
                    f"total images: {total_images:,}"
                )

                # Check if training is complete
                if iteration >= cfg.training.max_iterations - 1:
                    log.info("Training completed!")
                    if cfg.use_wandb:
                        wandb.finish()
                    return

                iteration += 1

    except Exception as e:
        log.error(f"Training interrupted: {e}")
        if cfg.use_wandb and wandb.run:
            wandb.finish()

        if iteration < cfg.training.max_iterations - 1:
            log.info("Saving checkpoint before exit...")
            save_dir = (
                Path(wandb.run.dir)
                if (cfg.use_wandb and wandb.run)
                else Path(cfg.save_path)
            )
            checkpoint_path = save_checkpoint(
                OmegaConf.to_container(cfg, resolve=True),
                network,
                optimizer,
                iteration + 1,
                save_dir,
                triplet_loss=triplet_loss_fn,
                random_states=seed_management("store"),
                save_all=True,
                total_images=total_images,
            )
            log.info(f"Saved checkpoint: {checkpoint_path}")

        raise


if __name__ == "__main__":
    main()
