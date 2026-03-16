import sys
import torch
import logging
import wandb
import time
import numpy as np
from typing import Optional, Union, Tuple
from pathlib import Path
import random

from omegaconf import DictConfig, OmegaConf

# Local imports
from datasets.dataset_loaders import (
    get_dataloaders,
)
from model.network_descriptor import SANDesc
from losses.triplet_loss import TripletLoss
from lr_scheduler.lr_scheduler import LrManager


# Get logger
log = logging.getLogger(__name__)


def setup_paths():
    """Add all necessary paths to sys.path"""
    project_root = Path(__file__).parent

    # Add project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add train_utils explicitly
    train_utils_path = project_root / "train_utils"
    if train_utils_path.exists() and str(train_utils_path) not in sys.path:
        sys.path.insert(0, str(train_utils_path))

    # Add libutils if it exists
    libutils_path = project_root / "libutils"
    if libutils_path.exists() and str(libutils_path) not in sys.path:
        sys.path.insert(0, str(libutils_path))

    # Add external repos
    external_repos = [
        "/home/mattia/Desktop/Repos/posebench",
    ]

    for repo_path in external_repos:
        if Path(repo_path).exists() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)


def sanitize_config_for_omegaconf(config_dict):
    """Convert non-serializable types to OmegaConf-compatible types"""
    sanitized = {}

    for key, value in config_dict.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_config_for_omegaconf(value)
        elif isinstance(value, torch.dtype):
            # Convert torch.dtype to string
            if value == torch.float16:
                sanitized[key] = "float16"
            elif value == torch.float32:
                sanitized[key] = "float32"
            elif value == torch.bfloat16:
                sanitized[key] = "bfloat16"
            else:
                sanitized[key] = str(value)
        elif hasattr(value, "__module__") and hasattr(value, "__name__"):
            # Convert other non-primitive types to strings
            sanitized[key] = str(value)
        else:
            sanitized[key] = value

    return sanitized


def setup_dtype_from_string(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype"""
    if isinstance(dtype_str, torch.dtype):
        return dtype_str  # Already a dtype

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def load_checkpoint_if_needed(resume_from: Optional[str]) -> Optional[dict]:
    """Load checkpoint if resume path is provided"""
    if resume_from is None:
        return None

    if not Path(resume_from).exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

    return torch.load(resume_from, weights_only=False)


def setup_dataloaders(cfg: DictConfig):
    """Setup dataloaders"""
    # Validation dataloader (normal)
    config_override = {
        "covisibility_weights": {
            "0.0": 0.0,
            "0.1": 0.0,
            "0.2": 0.0,
            "0.3": 0.1,
            "0.4": 0.1,
            "0.5": 0.1,
            "0.6": 0.1,
            "0.7": 0.1,
            "0.8": 0.1,
            "0.9": 0.1,
        }
    }

    _, valid_dataloader, compute_GT_matching_matrix_fn_validation, _ = get_dataloaders(
        "imb",
        1,
        cfg.model.unet_ch_in,
        config_override=config_override,
        num_workers=0,
    )

    # Validation dataloader (hard)
    config_override_hard = {
        "covisibility_weights": {
            "0.0": 0.0,
            "0.1": 0.1,
            "0.2": 0.1,
            "0.3": 0.0,
            "0.4": 0.0,
            "0.5": 0.0,
            "0.6": 0.0,
            "0.7": 0.0,
            "0.8": 0.0,
            "0.9": 0.0,
        }
    }

    _, valid_dataloader_hard, _, _ = get_dataloaders(
        "imb",
        1,
        cfg.model.unet_ch_in,
        config_override=config_override_hard,
        num_workers=0,
    )

    # Training dataloader
    random_rotation_degrees_fn = lambda: np.random.uniform(
        -cfg.training.random_training_rotation, cfg.training.random_training_rotation
    )

    train_dataloader, _, compute_GT_matching_matrix_fn, config_dataset = (
        get_dataloaders(
            cfg.training.dataset,
            cfg.training.batch_size,
            img_channels=cfg.model.unet_ch_in,
            num_workers=8,
            config_override={"random_rotation_degrees_fn": random_rotation_degrees_fn},
            augment=cfg.training.photo_aug_in_training,
        )
    )

    return (
        train_dataloader,
        valid_dataloader,
        valid_dataloader_hard,
        compute_GT_matching_matrix_fn,
        compute_GT_matching_matrix_fn_validation,
    )


def setup_model_and_optimizer(cfg: DictConfig, checkpoint: Optional[dict]):
    """Setup model and optimizer"""
    # Model - same parameters as your ModelConfig dataclass
    network = SANDesc(
        ch_in=cfg.model.unet_ch_in,
        kernel_size=cfg.model.unet_kernel_size,
        activ=cfg.model.unet_activ,
        norm=cfg.model.unet_norm,
        skip_connection=cfg.model.unet_with_skip_connections,
        spatial_attention=cfg.model.unet_spatial_attention,
        third_block=cfg.model.third_block,
    ).to(cfg.device)

    # Learning rate scheduler
    lr_scheduler = LrManager(
        name=cfg.training.lr_scheduler,
        lr_min=cfg.training.lr_min,
        lr_max=cfg.training.lr_max,
        decay_steps=cfg.training.decay_steps,
        max_iterations=cfg.training.max_iterations,
        warmup_steps=cfg.training.warmup_steps,
    )

    initial_lr = lr_scheduler.get_lr(0)
    optimizer = torch.optim.AdamW(network.parameters(), lr=initial_lr)

    # Resume from checkpoint
    network, optimizer, iteration = resume_from_checkpoint(
        network, optimizer, 0, checkpoint
    )

    return network, optimizer, lr_scheduler, iteration


def setup_loss_and_scaler(cfg: DictConfig):
    """Setup loss and scaler"""
    # Create TripletLoss with correct config path
    triplet_loss_fn = TripletLoss(
        margin=cfg.triplet_loss.margin,
        ratio=cfg.triplet_loss.ratio,
        random_negative_ratio=cfg.triplet_loss.random_negative_ratio,
        random_negative_ratio_decay=cfg.triplet_loss.random_negative_ratio_decay,
        verbose=cfg.triplet_loss.verbose,
        quadratic=cfg.triplet_loss.quadratic,
    )

    log.info("TripletLoss initialized with:")
    log.info(f"  - margin: {cfg.triplet_loss.margin}")
    log.info(f"  - ratio: {cfg.triplet_loss.ratio}")
    log.info(f"  - random_negative_ratio: {cfg.triplet_loss.random_negative_ratio}")
    log.info(
        f"  - random_negative_ratio_decay: {cfg.triplet_loss.random_negative_ratio_decay}"
    )
    log.info(f"  - verbose: {cfg.triplet_loss.verbose}")

    # Setup AMP
    amp_device_type = "cuda"
    amp_dtype = setup_dtype_from_string(cfg.training.amp_dtype)
    scaler = torch.amp.GradScaler(amp_device_type, enabled=cfg.training.use_amp)

    return triplet_loss_fn, scaler, amp_device_type, amp_dtype


def setup_wrappers(cfg: DictConfig):
    """Setup keypoint detection wrappers"""
    try:
        # this is imported from set up paths
        from wrappers_manager import wrappers_manager
    except ImportError:
        logging.warning(
            "Could not import wrappers_manager. Install it from github.com/mattiadurso/PoseBench"
        )
        exit()

    train_wrapper = wrappers_manager(cfg.training.train_wrapper, cfg.device)

    if cfg.training.valid_wrapper == cfg.training.train_wrapper:
        valid_wrapper = train_wrapper
    else:
        valid_wrapper = wrappers_manager(cfg.training.valid_wrapper, cfg.device)

    return train_wrapper, valid_wrapper


def setup_logging(cfg: DictConfig, checkpoint: Optional[dict]):
    """Setup logging"""
    if cfg.use_wandb:
        if checkpoint is None:
            run_name = f"{cfg.training.train_wrapper}+{cfg.training.run_name}"
            wandb.init(
                project="sandesc",
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir="./",
                tags=["paper"],
                settings=wandb.Settings(code_dir="."),
            )
        else:
            wandb.init(
                project="sandesc",
                name=f"{cfg.training.run_name}_from_{checkpoint['iteration']}it",
                config=OmegaConf.to_container(cfg, resolve=True),
                dir="./",
                tags=["paper"],
                settings=wandb.Settings(code_dir="."),
                resume="allow",
                id=cfg.training.run_id,
            )

        wandb.run.log_code(".")
        cfg.training.run_id = wandb.run.id


def set_deterministic_behavior():
    """Set deterministic behavior for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def resume_from_checkpoint(network, optimizer, iteration, checkpoint=None):
    """
    Resume from checkpoint
    """
    if checkpoint is None:
        print("No checkpoint provided")
        return network, optimizer, iteration

    print(f"Loading checkpoint...")

    # resuming network
    network.load_state_dict(checkpoint["state_dict"])

    # resuming optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # resuming iteration
    iteration = checkpoint["iteration"]

    print(
        f'Resumed from iteration: {iteration:,}/{checkpoint["config"]["training"]["max_iterations"]:,}'
    )

    return network, optimizer, iteration


def compute_grad_norm(network):
    grad_norm = 0.0
    for p in network.parameters():
        if p.grad is not None:
            grad = p.grad.detach()

            param_norm = grad.norm(2)  # L2 norm
            grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm**0.5  # Final L2 norm over all parameters
    return grad_norm


def seed_management(
    mode: str, values: Union[None, Tuple, int] = None
) -> Union[None, Tuple]:
    """random seed management function
    Args:
         mode: can be 'store', 'store_and_reset', 'restore', 'reset'
         values: can be either the saved randoms states tuple or the seed
    Returns:
        Union[None, Tuple]: current randoms states tuple if mode is 'store' or 'store_and_reset', else is None
    Raises:
        None
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    if mode == "store_and_reset":
        # save the current random states and reset the state
        torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_random_state = torch.cuda.get_rng_state()
            torch_cuda_all_random_state = torch.cuda.get_rng_state_all()
        else:
            torch_cuda_random_state = 0
            torch_cuda_all_random_state = 0
        np_random_state = np.random.get_state()
        python_random_state = random.getstate()
        seed = values if values is not None else 0
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        np.random.seed(seed)
        random.seed(seed)

        return (
            torch_random_state,
            torch_cuda_random_state,
            torch_cuda_all_random_state,
            np_random_state,
            python_random_state,
        )

    elif mode == "store":
        torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_random_state = torch.cuda.get_rng_state()
            torch_cuda_all_random_state = torch.cuda.get_rng_state_all()
        else:
            torch_cuda_random_state = 0
            torch_cuda_all_random_state = 0
        np_random_state = np.random.get_state()
        python_random_state = random.getstate()

        return (
            torch_random_state,
            torch_cuda_random_state,
            torch_cuda_all_random_state,
            np_random_state,
            python_random_state,
        )

    elif mode == "restore" and values is not None:
        # restore the random states
        (
            torch_random_state,
            torch_cuda_random_state,
            torch_cuda_all_random_state,
            np_random_state,
            python_random_state,
        ) = values
        torch.random.set_rng_state(torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random_state)
            torch.cuda.set_rng_state_all(torch_cuda_all_random_state)
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        print("Random states restored correctly")

    elif mode == "reset":
        # set all the seeds
        seed = values if values is not None else 0
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
        np.random.seed(seed)
        random.seed(seed)
    else:
        raise ValueError
