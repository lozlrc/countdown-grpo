import math

import pytest
import torch

from grpo.loss import grpo_loss


def make_inputs():
    """2 sequences, 4 positions; first two positions are prompt."""
    old = torch.tensor(
        [
            [0.0, -1.0, -1.0, -2.0],
            [0.0, -1.0, -0.5, -1.5],
        ]
    )
    new = old.clone()
    mask = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    adv = torch.tensor([1.0, -0.5])
    return new, old, adv, mask


def test_ratio_one_gives_negative_advantage_mean():
    new, old, adv, mask = make_inputs()
    loss, stats = grpo_loss(new, old, adv, mask)
    # ratio == 1 everywhere: per-token loss is -A, token_mean over 4 tokens
    expected = -(1.0 + 1.0 - 0.5 - 0.5) / 4
    assert loss.item() == pytest.approx(expected)
    assert stats["clip_frac"] == 0.0


def test_prompt_tokens_contribute_nothing():
    new, old, adv, mask = make_inputs()
    baseline, _ = grpo_loss(new, old, adv, mask)
    corrupted = new.clone()
    corrupted[:, :2] = 123.0  # garbage at masked (prompt) positions
    perturbed, _ = grpo_loss(corrupted, old, adv, mask)
    assert perturbed.item() == pytest.approx(baseline.item())


def test_prompt_tokens_get_no_gradient():
    new, old, adv, mask = make_inputs()
    leaf = new.clone().requires_grad_(True)
    loss, _ = grpo_loss(leaf, old, adv, mask)
    loss.backward()
    assert torch.equal(leaf.grad[:, :2], torch.zeros(2, 2))
    assert (leaf.grad[:, 2:] != 0).all()


@pytest.mark.parametrize(
    "log_ratio,adv,expected",
    [
        # positive advantage, ratio 2.0 > 1+eps: clipped to 1.2
        (math.log(2.0), 1.0, -1.2),
        # negative advantage, ratio 2.0: min(-2.0, -1.2) takes unclipped
        (math.log(2.0), -1.0, 2.0),
        # positive advantage, ratio 0.5 < 1-eps: min(0.5, 0.8) unclipped
        (math.log(0.5), 1.0, -0.5),
        # negative advantage, ratio 0.5: clipped at 0.8
        (math.log(0.5), -1.0, 0.8),
        # ratio exactly at the boundary: clip is a no-op
        (math.log(1.2), 1.0, -1.2),
    ],
)
def test_clip_boundaries(log_ratio, adv, expected):
    old = torch.tensor([[0.0, -1.0]])
    new = torch.tensor([[0.0, -1.0 + log_ratio]])
    mask = torch.tensor([[0.0, 1.0]])
    loss, _ = grpo_loss(new, old, torch.tensor([adv]), mask, clip_eps=0.2)
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_kl_zero_equals_no_ref():
    new, old, adv, mask = make_inputs()
    ref = old - 0.3
    loss_no_ref, _ = grpo_loss(new, old, adv, mask, kl_beta=0.0)
    loss_with_ref, _ = grpo_loss(new, old, adv, mask, kl_beta=0.0, ref_logprobs=ref)
    assert loss_no_ref.item() == loss_with_ref.item()


def test_kl_at_reference_is_zero():
    new, old, adv, mask = make_inputs()
    loss_base, _ = grpo_loss(new, old, adv, mask)
    loss_kl, stats = grpo_loss(new, old, adv, mask, kl_beta=0.1, ref_logprobs=new)
    assert loss_kl.item() == pytest.approx(loss_base.item())
    assert stats["kl_ref"] == pytest.approx(0.0)


def test_kl_penalty_hand_computed():
    old = torch.tensor([[0.0, -1.0]])
    new = torch.tensor([[0.0, -1.0]])
    ref = torch.tensor([[0.0, -1.5]])
    mask = torch.tensor([[0.0, 1.0]])
    adv = torch.tensor([0.0])
    beta = 0.5
    # k3 estimator: exp(ref - new) - (ref - new) - 1 with ref - new = -0.5
    k3 = math.exp(-0.5) + 0.5 - 1.0
    loss, stats = grpo_loss(new, old, adv, mask, kl_beta=beta, ref_logprobs=ref)
    assert loss.item() == pytest.approx(beta * k3, abs=1e-6)
    assert stats["kl_ref"] == pytest.approx(k3, abs=1e-6)
    assert k3 > 0


def test_kl_requires_ref():
    new, old, adv, mask = make_inputs()
    with pytest.raises(ValueError):
        grpo_loss(new, old, adv, mask, kl_beta=0.1)


def test_agg_modes():
    # seq 0 has one action token, seq 1 has three
    old = torch.tensor(
        [
            [0.0, -1.0, -1.0, -1.0],
            [0.0, -1.0, -1.0, -1.0],
        ]
    )
    new = old.clone()
    mask = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    adv = torch.tensor([1.0, -1.0])
    token_mean, _ = grpo_loss(new, old, adv, mask, agg="token_mean")
    seq_mean, _ = grpo_loss(new, old, adv, mask, agg="seq_mean")
    drgrpo, _ = grpo_loss(new, old, adv, mask, agg="drgrpo", max_tokens=8)
    # token_mean: (-1 + 3) / 4 tokens; seq_mean: (-1 + 1) / 2; drgrpo: 2 / 16
    assert token_mean.item() == pytest.approx(0.5)
    assert seq_mean.item() == pytest.approx(0.0)
    assert drgrpo.item() == pytest.approx(2.0 / 16.0)


def test_drgrpo_requires_max_tokens():
    new, old, adv, mask = make_inputs()
    with pytest.raises(ValueError):
        grpo_loss(new, old, adv, mask, agg="drgrpo")


def test_unknown_agg():
    new, old, adv, mask = make_inputs()
    with pytest.raises(ValueError):
        grpo_loss(new, old, adv, mask, agg="mean")
