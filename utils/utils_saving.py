import os
from typing import Dict
from pathlib import Path

import torch as th
from torch import nn as nn


def save_checkpoint(
    config,
    model: nn.Module,
    optimizer: th.optim.Optimizer,
    iteration: int,
    save_path: Path = None,
    triplet_loss=None,
    random_states=None,
    save_all=False,
    total_images: int = 0,
) -> None:
    """
    Saving checkpoints
    """

    arch = type(model).__name__
    try:
        config = config.as_dict()
    except:
        pass

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
        os.makedirs(save_path / "saved_model", exist_ok=True)
        checkpoint_path = (
            save_path / "saved_model" / f"checkpoint-iteration-{iteration}.pth"
        )
    else:
        checkpoint_path = save_path / "saved_model" / "checkpoint.pth"

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    th.save(state, checkpoint_path)  # ? save everything

    print(f"Checkpoint saved at {checkpoint_path}, iteration: {iteration}")
    return checkpoint_path


def resume_checkpoint(
    model: nn.Module, optimizer: th.optim.Optimizer, checkpoint: Dict
) -> None:
    try:
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except RuntimeError:
        print(
            "WARNING: current architecture and model loaded do NOT correspond, loading with <strict=False>"
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"], strict=False)
