"""Rollout logprob computation on the tiny random model (CPU, fast)."""

import pytest
import torch
import torch.nn.functional as F

from grpo.config import GenCfg
from grpo.models import build_char_tokenizer, build_tiny_model
from grpo.rollout import sample_rollouts, token_logprobs


@pytest.fixture(scope="module")
def tiny():
    tokenizer = build_char_tokenizer()
    model = build_tiny_model(len(tokenizer), seed=0).eval()
    return model, tokenizer


def test_token_logprobs_apply_temperature(tiny):
    model, tokenizer = tiny
    enc = tokenizer(["12 + 7 =", "count: 3"], return_tensors="pt", padding=True)
    ids, attn = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    for temp in (1.0, 2.0):
        got = token_logprobs(model, ids, attn, temperature=temp)
        want = (
            F.log_softmax(logits / temp, dim=-1)
            .gather(-1, ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.equal(got[:, 0], torch.zeros(ids.shape[0]))
        assert torch.allclose(got[:, 1:], want, atol=1e-6)


def test_old_logprobs_match_sampling_temperature(tiny):
    model, tokenizer = tiny
    torch.manual_seed(0)
    gen = GenCfg(group_size=2, temperature=2.0, top_p=1.0, max_new_tokens=8)
    batch = sample_rollouts(model, tokenizer, ["12 + 7 ="], gen, torch.device("cpu"))
    tempered = token_logprobs(
        model, batch.sequences, batch.attention_mask, temperature=2.0
    )
    untempered = token_logprobs(model, batch.sequences, batch.attention_mask)
    mask = batch.action_mask.bool()
    assert torch.allclose(batch.old_logprobs[mask], tempered[mask], atol=1e-5)
    # sampled tokens must be scored under the tempered policy, not raw logits
    assert not torch.allclose(batch.old_logprobs[mask], untempered[mask], atol=1e-3)


def test_sample_rollouts_rejects_top_p(tiny):
    model, tokenizer = tiny
    gen = GenCfg(group_size=2, temperature=1.0, top_p=0.9, max_new_tokens=8)
    with pytest.raises(ValueError, match="top_p"):
        sample_rollouts(model, tokenizer, ["12 + 7 ="], gen, torch.device("cpu"))
