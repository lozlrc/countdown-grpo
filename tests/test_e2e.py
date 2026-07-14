"""End-to-end policy improvement test.

Runs the full GRPO loop (sample -> reward -> group advantages -> clipped
update) on a tiny random model with a rigged reward: +1 for any completion
containing the character "z". If the machinery is correct, the policy's
probability of emitting "z" must increase measurably. This is the strongest
correctness check in the suite: it exercises rollout, masking, old-logprob
bookkeeping, advantage normalization, and the surrogate loss together.
"""

import torch

from grpo.advantage import group_advantages
from grpo.config import Config
from grpo.models import build_char_tokenizer, build_tiny_model
from grpo.rollout import sample_rollouts
from grpo.train import grpo_update

REWARD_CHAR = "z"
PROMPTS = ["12 + 7 =", "solve x:", "count:", "answer ->"]


def make_config() -> Config:
    cfg = Config()
    cfg.gen.group_size = 8
    cfg.gen.max_new_tokens = 16
    cfg.gen.temperature = 1.0
    cfg.gen.top_p = 1.0
    cfg.train.minibatch_size = 16
    cfg.train.epochs_per_batch = 1
    cfg.train.grad_accum = 1
    cfg.loss.clip_eps = 0.2
    cfg.loss.kl_beta = 0.0
    cfg.loss.agg = "token_mean"
    cfg.optim.grad_clip = 1.0
    return cfg


@torch.no_grad()
def reward_char_prob(model, tokenizer, char_id: int) -> float:
    """Mean next-token probability of the rewarded character over the
    prompt positions — a low-variance measure of the policy's shift."""
    model.eval()
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
    probs = logits.float().softmax(dim=-1)[..., char_id]
    return probs[enc["attention_mask"].bool()].mean().item()


def test_grpo_increases_rewarded_token_probability():
    torch.manual_seed(0)
    device = torch.device("cpu")
    tokenizer = build_char_tokenizer()
    char_id = tokenizer.convert_tokens_to_ids(REWARD_CHAR)
    assert char_id != tokenizer.unk_token_id

    model = build_tiny_model(len(tokenizer), seed=0).to(device)
    cfg = make_config()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    prob_before = reward_char_prob(model, tokenizer, char_id)

    reward_rates = []
    for _ in range(30):
        batch = sample_rollouts(model, tokenizer, PROMPTS, cfg.gen, device)
        rewards = torch.tensor(
            [1.0 if REWARD_CHAR in c else 0.0 for c in batch.completions]
        )
        reward_rates.append(rewards.mean().item())
        advantages, _ = group_advantages(rewards, cfg.gen.group_size)
        grpo_update(model, optimizer, batch, advantages, cfg)

    prob_after = reward_char_prob(model, tokenizer, char_id)

    # the policy must have learned to emit the rewarded token far more often
    assert prob_after > 3 * prob_before, (prob_before, prob_after)
    assert prob_after > 0.05, (prob_before, prob_after)

    early = sum(reward_rates[:5]) / 5
    late = sum(reward_rates[-5:]) / 5
    assert late > early, (early, late)
