"""Tests for the LrManager learning-rate schedules."""

import pytest
import torch

from lr_scheduler.lr_scheduler import LrManager


def test_constant_returns_lr_min() -> None:
    """The 'constant' schedule always returns lr_min."""
    lr = LrManager(name="constant", lr_min=1e-4, lr_max=5e-3)
    assert lr.get_lr(0) == pytest.approx(1e-4)
    assert lr.get_lr(10_000) == pytest.approx(1e-4)


def test_warmup_decay_warmup_phase_is_monotonic() -> None:
    """During warmup the LR rises from ~lr_min to lr_max."""
    lr = LrManager(
        name="warmup_decay_constant", lr_min=1e-4, lr_max=5e-3, warmup_steps=100
    )
    assert lr.get_lr(0) == pytest.approx(1e-4)
    assert lr.get_lr(100) == pytest.approx(5e-3)
    warmup = [lr.get_lr(i) for i in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(warmup, warmup[1:]))


def test_warmup_decay_never_below_lr_min() -> None:
    """After decay the LR is floored at lr_min."""
    lr = LrManager(
        name="warmup_decay_constant", lr_min=1e-4, lr_max=5e-3, warmup_steps=100
    )
    assert lr.get_lr(10_000_000) == pytest.approx(1e-4)
    assert lr.get_lr(500) > lr.get_lr(5000)  # decaying after warmup


def test_unknown_schedule_raises() -> None:
    """An unknown schedule name raises ValueError."""
    lr = LrManager(name="does_not_exist")
    with pytest.raises(ValueError, match="Unknown lr_scheduler"):
        lr.get_lr(0)


def test_update_lr_sets_optimizer_param_groups() -> None:
    """update_lr writes the computed LR onto every optimizer param group."""
    lr = LrManager(name="constant", lr_min=2e-4)
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    lr.update_lr(optimizer, iteration=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
