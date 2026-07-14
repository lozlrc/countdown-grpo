"""Batched rollout sampling and per-token logprob computation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from grpo.config import GenCfg


@dataclass
class RolloutBatch:
    """G completions per prompt, flattened to N = num_prompts * G rows.

    Rows are grouped: rows [i*G, (i+1)*G) all belong to prompt i, which is
    what advantage.group_advantages expects.
    """

    sequences: torch.Tensor  # [N, T] left-padded prompt + completion ids
    attention_mask: torch.Tensor  # [N, T] 1 on real tokens up to first EOS
    action_mask: torch.Tensor  # [N, T] 1 on completion tokens up to first EOS
    old_logprobs: torch.Tensor  # [N, T] logprobs under the sampling policy
    completions: list[str]  # decoded completion strings
    prompt_len: int  # width of the left-padded prompt block
    group_size: int

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            sequences=self.sequences.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_logprobs=self.old_logprobs.to(device),
            completions=self.completions,
            prompt_len=self.prompt_len,
            group_size=self.group_size,
        )


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token logprob of input_ids[:, t] given the prefix, [N, T].

    temperature must match the sampling temperature: generate draws from
    softmax(logits / T), so scoring raw logits against those samples would
    silently bias the surrogate off-policy for any T != 1.

    Column 0 has no prediction and is set to 0; it is always a prompt token
    and masked out downstream.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :].float() / temperature
    targets = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    logps = logps.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    zeros = logps.new_zeros(logps.shape[0], 1)
    return torch.cat([zeros, logps], dim=1)


def batched_token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    minibatch_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """token_logprobs in no-grad minibatches (keeps peak memory bounded)."""
    outs = []
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], minibatch_size):
            outs.append(
                token_logprobs(
                    model,
                    input_ids[i : i + minibatch_size],
                    attention_mask[i : i + minibatch_size],
                    temperature=temperature,
                )
            )
    return torch.cat(outs, dim=0)


def completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """1.0 on tokens up to and including the first EOS, 0.0 after."""
    is_eos = completion_ids == eos_token_id
    after_eos = (is_eos.cumsum(dim=1) - is_eos.long()) > 0
    return (~after_eos).float()


@torch.no_grad()
def sample_rollouts(
    model,
    tokenizer,
    prompts: list[str],
    gen_cfg: GenCfg,
    device: torch.device,
    logprob_minibatch: int = 8,
) -> RolloutBatch:
    """Sample group_size completions per prompt and score old logprobs."""
    if gen_cfg.top_p != 1.0:
        raise ValueError(
            "top_p != 1.0 truncates the sampling distribution in a way the "
            "logprob computation cannot account for; use temperature instead"
        )
    was_training = model.training
    model.eval()

    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        do_sample=True,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        max_new_tokens=gen_cfg.max_new_tokens,
        num_return_sequences=gen_cfg.group_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completion_ids = sequences[:, prompt_len:]
    comp_mask = completion_mask(completion_ids, tokenizer.eos_token_id)

    # generate returns rows grouped per input prompt
    prompt_attn = attn.repeat_interleave(gen_cfg.group_size, dim=0)
    attention_mask = torch.cat([prompt_attn.float(), comp_mask], dim=1)

    action_mask = torch.cat(
        [torch.zeros_like(prompt_attn, dtype=torch.float32), comp_mask], dim=1
    )

    old_logprobs = batched_token_logprobs(
        model,
        sequences,
        attention_mask,
        logprob_minibatch,
        temperature=gen_cfg.temperature,
    )

    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    if was_training:
        model.train()
    return RolloutBatch(
        sequences=sequences,
        attention_mask=attention_mask,
        action_mask=action_mask,
        old_logprobs=old_logprobs,
        completions=completions,
        prompt_len=prompt_len,
        group_size=gen_cfg.group_size,
    )
