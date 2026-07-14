"""Token-level PPO-clip surrogate with optional KL penalty against a
frozen reference model (GRPO objective, DeepSeekMath eq. 3)."""

from __future__ import annotations

import torch

AGG_MODES = ("token_mean", "seq_mean", "drgrpo")


def grpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
    ref_logprobs: torch.Tensor | None = None,
    agg: str = "token_mean",
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO loss over response tokens only.

    logprobs / old_logprobs / ref_logprobs: [N, T] per-token logprobs of the
    sampled sequence (position t = logprob of token t given prefix).
    advantages: [N] per-sequence group-relative advantages.
    action_mask: [N, T] 1.0 on completion tokens, 0.0 on prompt/padding.
    Prompt tokens are excluded from the objective entirely via the mask.

    kl_beta=0 disables the KL term and no reference model is needed
    (GRPO-Zero mode). Otherwise KL uses the low-variance k3 estimator from
    the GRPO paper: exp(ref - pi) - (ref - pi) - 1, which is >= 0.

    agg controls masked-mean normalization (the Dr. GRPO knob):
      token_mean: sum over all response tokens / total response tokens.
      seq_mean:   per-sequence token mean, then mean over sequences
                  (original GRPO 1/|o_i| weighting; biases toward long
                  wrong answers).
      drgrpo:     sum / (N * max_tokens), a constant normalizer that removes
                  the length bias (requires max_tokens).
    """
    if agg not in AGG_MODES:
        raise ValueError(f"agg must be one of {AGG_MODES}")
    if agg == "drgrpo" and max_tokens is None:
        raise ValueError("agg='drgrpo' requires max_tokens")
    if kl_beta > 0.0 and ref_logprobs is None:
        raise ValueError("kl_beta > 0 requires ref_logprobs")

    mask = action_mask.float()
    adv = advantages.unsqueeze(1)  # [N, 1] broadcast over tokens

    # Zero the log-ratio at masked positions BEFORE exponentiating: garbage
    # logprobs at prompt/padding positions would otherwise produce inf * 0
    # = nan when the mask is applied after.
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token = -torch.minimum(unclipped, clipped)

    kl_value = 0.0
    if kl_beta > 0.0:
        ref_log_ratio = (ref_logprobs - logprobs) * mask
        kl = ref_log_ratio.exp() - ref_log_ratio - 1.0
        per_token = per_token + kl_beta * kl
        kl_value = _masked_scalar(kl, mask)

    if agg == "token_mean":
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    elif agg == "seq_mean":
        per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = per_seq.mean()
    else:  # drgrpo
        loss = (per_token * mask).sum() / (per_token.shape[0] * max_tokens)

    with torch.no_grad():
        # clipping binds only where min() selects the clipped branch (e.g.
        # A < 0 with ratio > 1+eps still takes the unclipped term)
        clip_frac = _masked_scalar((clipped < unclipped).float(), mask)
        # k2-style estimate of KL(pi_theta || pi_old), for monitoring drift
        approx_kl = _masked_scalar((ratio - 1.0) - log_ratio, mask)

    stats = {
        "loss": loss.item(),
        "clip_frac": clip_frac,
        "approx_kl_old": approx_kl,
        "kl_ref": kl_value,
    }
    return loss, stats


def _masked_scalar(x: torch.Tensor, mask: torch.Tensor) -> float:
    return ((x * mask).sum() / mask.sum().clamp(min=1.0)).item()
