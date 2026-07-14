import math

import pytest
import torch

from grpo.advantage import group_advantages


def test_hand_computed_single_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv, degenerate = group_advantages(rewards, group_size=4)
    # mean 0.5, population std 0.5
    expected = (rewards - 0.5) / (0.5 + 1e-6)
    assert torch.allclose(adv, expected)
    assert not degenerate.any()


def test_hand_computed_uneven_group():
    rewards = torch.tensor([1.0, 0.1, 0.0, 0.0])
    mean = 0.275
    std = math.sqrt(sum((r - mean) ** 2 for r in [1.0, 0.1, 0.0, 0.0]) / 4)
    adv, _ = group_advantages(rewards, group_size=4)
    expected = torch.tensor([(r - mean) / (std + 1e-6) for r in [1.0, 0.1, 0.0, 0.0]])
    assert torch.allclose(adv, expected, atol=1e-5)


def test_zero_variance_group_guarded():
    for value in (0.0, 1.0):
        rewards = torch.full((8,), value)
        adv, degenerate = group_advantages(rewards, group_size=8)
        assert torch.equal(adv, torch.zeros(8))
        assert degenerate.all()


def test_mixed_groups():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    adv, degenerate = group_advantages(rewards, group_size=4)
    assert torch.equal(adv[:4], torch.zeros(4))
    assert degenerate[:4].all() and not degenerate[4:].any()
    assert torch.allclose(adv[4:], (rewards[4:] - 0.5) / (0.5 + 1e-6))


def test_dr_grpo_no_std_normalization():
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv, _ = group_advantages(rewards, group_size=4, use_std=False)
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_dr_grpo_zero_variance_still_zero():
    adv, degenerate = group_advantages(torch.ones(4), group_size=4, use_std=False)
    assert torch.equal(adv, torch.zeros(4))
    assert degenerate.all()


def test_group_mean_is_zero():
    torch.manual_seed(0)
    rewards = torch.rand(32)
    adv, _ = group_advantages(rewards, group_size=8)
    assert torch.allclose(adv.view(4, 8).sum(dim=1), torch.zeros(4), atol=1e-5)


def test_shape_validation():
    with pytest.raises(ValueError):
        group_advantages(torch.ones(5), group_size=4)
    with pytest.raises(ValueError):
        group_advantages(torch.ones(4, 2), group_size=4)
