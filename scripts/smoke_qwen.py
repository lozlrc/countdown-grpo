"""One-off real-model smoke test: a single rollout batch with real weights.

Downloads Qwen2.5-0.5B (~1GB) on first run. Verifies that generation,
masking, reward extraction, and logprob computation behave with a real
tokenizer and model on the local device (MPS on Apple Silicon).

    uv run python scripts/smoke_qwen.py
    uv run python scripts/smoke_qwen.py --model Qwen/Qwen2.5-0.5B-Instruct --template chat
"""

from __future__ import annotations

import argparse
import time

import torch

from grpo.config import GenCfg, ModelCfg
from grpo.countdown import make_dataset, render_prompt, score_completion
from grpo.models import load_model_and_tokenizer, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--template", default="r1", choices=["plain", "chat", "r1"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--num-puzzles", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    from grpo.rollout import sample_rollouts

    device = pick_device(args.device)
    print(f"device: {device}")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(
        ModelCfg(name=args.model, dtype="auto", attn_implementation="sdpa"), device
    )
    print(f"loaded {args.model} in {time.time() - t0:.1f}s")

    puzzles = make_dataset(args.num_puzzles, seed=42)
    rendered = [render_prompt(p, args.template, tokenizer) for p in puzzles]
    prompts = [r[0] for r in rendered]
    think_open = rendered[0][1]

    gen_cfg = GenCfg(
        group_size=args.group_size,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=args.max_new_tokens,
    )
    t0 = time.time()
    batch = sample_rollouts(model, tokenizer, prompts, gen_cfg, device, logprob_minibatch=2)
    n = batch.sequences.shape[0]
    print(f"sampled {n} completions in {time.time() - t0:.1f}s")

    scores = [
        score_completion(c, puzzles[i // args.group_size], think_open=think_open)
        for i, c in enumerate(batch.completions)
    ]
    lens = batch.action_mask.sum(dim=1)
    mean_lp = (
        (batch.old_logprobs * batch.action_mask).sum() / batch.action_mask.sum()
    ).item()
    print(f"reward mean: {sum(s.reward for s in scores) / n:.3f}")
    print(f"format rate: {sum(s.format_ok for s in scores) / n:.3f}")
    print(f"answer rate: {sum(s.answer_ok for s in scores) / n:.3f}")
    print(f"completion len mean: {lens.float().mean().item():.0f}")
    print(f"mean action logprob: {mean_lp:.3f}")

    best = max(range(n), key=lambda i: (scores[i].reward, -lens[i].item()))
    p = puzzles[best // args.group_size]
    print("\n--- sample transcript " + "-" * 40)
    print(f"puzzle: nums={list(p.nums)} target={p.target} (known solution: {p.solution})")
    print(f"score: {scores[best]}")
    print(prompts[best // args.group_size], end="")
    print(batch.completions[best])


if __name__ == "__main__":
    main()
