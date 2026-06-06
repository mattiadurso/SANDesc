"""Checkpoint saving and resuming utilities."""

import contextlib
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn as nn

from losses.triplet_loss import TripletLoss

log = logging.getLogger(__name__)


def save_checkpoint(
    config: Any,  # noqa: ANN401 - Hydra/omegaconf config or plain dict
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    save_path: Path | None = None,
    triplet_loss: TripletLoss | None = None,
    random_states: tuple | None = None,
    save_all: bool = False,
    total_images: int = 0,
) -> Path:
    """Saving checkpoints."""
    arch = type(model).__name__
    with contextlib.suppress(AttributeError):
        config = config.as_dict()

    config["triplet_loss"]["random_negative_ratio"] = triplet_loss.random_negative_ratio

    state = {
        "arch": arch,
        "iteration": iteration,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "random_state": random_states,
        "total_images": total_images,
    }

    if save_all:
        (save_path / "saved_model").mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            save_path / "saved_model" / f"checkpoint-iteration-{iteration}.pth"
        )
    else:
        checkpoint_path = save_path / "saved_model" / "checkpoint.pth"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, checkpoint_path)  # save everything

    log.info("Checkpoint saved at %s, iteration: %s", checkpoint_path, iteration)
    return checkpoint_path


def resume_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint: dict
) -> None:
    """Load model and optimizer state from a checkpoint, falling back to non-strict."""
    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError:
        log.warning(
            "current architecture and model loaded do NOT correspond, "
            "loading with <strict=False>"
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"], strict=False)
