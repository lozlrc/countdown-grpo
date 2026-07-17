"""Measure the base model's Countdown solve rate by puzzle difficulty.

Motivation: GRPO's learning signal is gated on sampling finding correct
answers — a group carries a solve gradient only if it contains at least one
(and not G) correct completions. The 1.5B baseline run stayed flat at the base
model's ~1-2% solve rate on the (3,4)-number mix. If easier slices of the task
have a materially higher base solve rate, a curriculum (train on those first)
densifies the reward signal for free.

This script samples the base model on puzzle slices of increasing difficulty
and reports, per slice:
  pass@1     mean per-completion answer rate (what training sees per sample)
  pass@G     fraction of puzzles where >=1 of G samples is correct
  signal@G   fraction of puzzles where 0 < #correct < G  (= the fraction of
             GRPO groups that would carry a nonzero solve gradient)

Runs inference-only on CPU/MPS (fp16 on MPS). No training.

    uv run python scripts/solve_rate_by_difficulty.py \
        --model Qwen/Qwen2.5-1.5B --puzzles 48 --group-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from grpo.config import ModelCfg
from grpo.countdown import make_dataset, render_prompt, score_completion
from grpo.models import load_model_and_tokenizer, pick_device

SLICES = [
    # (label, make_dataset kwargs)
    ("3num_small", dict(num_counts=(3,), nums_min=1, nums_max=30)),
    ("3num", dict(num_counts=(3,), nums_min=1, nums_max=99)),
    ("4num", dict(num_counts=(4,), nums_min=1, nums_max=99)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--puzzles", type=int, default=48, help="puzzles per slice")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/solve_rate_by_difficulty.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device("auto")
    # bf16 on MPS: fp16 overflows Qwen logits into NaN during sampling on Apple GPUs
    dtype = "bfloat16" if device.type == "mps" else "auto"
    print(f"device={device} dtype={dtype} model={args.model}", flush=True)
    model, tokenizer = load_model_and_tokenizer(ModelCfg(name=args.model, dtype=dtype), device)
    model.eval()

    results = {}
    for label, kwargs in SLICES:
        puzzles = make_dataset(args.puzzles, seed=args.seed + 1, **kwargs)
        n_correct_per_puzzle: list[int] = []
        fmt_hits = 0
        t0 = time.time()
        for start in range(0, len(puzzles), args.batch_prompts):
            chunk = puzzles[start : start + args.batch_prompts]
            prompts = [render_prompt(p, "r1")[0] for p in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=1.0,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.group_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            comps = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for i, p in enumerate(chunk):
                correct = 0
                for g in range(args.group_size):
                    s = score_completion(comps[i * args.group_size + g], p, think_open=True)
                    correct += s.answer_ok
                    fmt_hits += s.format_ok
                n_correct_per_puzzle.append(correct)
            done = start + len(chunk)
            print(f"  [{label}] {done}/{len(puzzles)} puzzles ({time.time()-t0:.0f}s)", flush=True)

        G = args.group_size
        n = len(puzzles)
        total = n * G
        res = {
            "pass_at_1": sum(n_correct_per_puzzle) / total,
            "pass_at_G": sum(c > 0 for c in n_correct_per_puzzle) / n,
            "signal_at_G": sum(0 < c < G for c in n_correct_per_puzzle) / n,
            "format_rate": fmt_hits / total,
            "n_puzzles": n,
            "group_size": G,
        }
        results[label] = res
        print(f"[{label}] pass@1 {res['pass_at_1']:.3f} | pass@{G} {res['pass_at_G']:.3f} | "
              f"signal@{G} {res['signal_at_G']:.3f} | format {res['format_rate']:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "temperature": args.temperature,
                               "max_new_tokens": args.max_new_tokens, "slices": results}, indent=2))
    print(f"\nwrote {out}")
    print("\n=== curriculum verdict ===")
    for label, r in results.items():
        print(f"  {label:>10}: a GRPO group would carry solve-signal {r['signal_at_G']*100:.0f}% of the time")


if __name__ == "__main__":
    main()
