"""Group-relative advantage estimation."""

from __future__ import annotations

import torch


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    use_std: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-sequence advantages relative to the sequence's group.

    rewards is a flat [N] tensor where N = num_groups * group_size and
    consecutive blocks of group_size share a prompt.

    use_std=True is standard GRPO: A = (r - mean) / (std + eps). When every
    reward in a group is identical, std is 0 and (r - mean) is also 0, so the
    naive formula is a 0/0 up to eps; we set those advantages to exactly 0
    and flag the group as degenerate so the caller can drop it (it carries no
    learning signal either way).

    use_std=False is the Dr. GRPO variant (arXiv:2503.20783): mean-centering
    only, which removes the difficulty bias introduced by std scaling.

    Returns (advantages [N], degenerate [N] bool mask of zero-variance groups).
    """
    if rewards.dim() != 1:
        raise ValueError("rewards must be a flat [N] tensor")
    if rewards.numel() % group_size != 0:
        raise ValueError("rewards length must be a multiple of group_size")

    grouped = rewards.view(-1, group_size).float()
    mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - mean
    degenerate = (grouped.std(dim=1, unbiased=False, keepdim=True) < eps).expand_as(
        grouped
    )

    if use_std:
        std = grouped.std(dim=1, unbiased=False, keepdim=True)
        adv = centered / (std + eps)
    else:
        adv = centered
    adv = torch.where(degenerate, torch.zeros_like(adv), adv)
    return adv.reshape(-1), degenerate.reshape(-1)
