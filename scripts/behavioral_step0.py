"""Step-0 behavioral analysis: is the 'aha moment' already in the base model?

Before a single gradient step, we sample the *base* model on Countdown with the
r1 template and count cognitive-behavior language -- verification, backtracking,
and enumeration -- in what it generates. If these phrases are already common at
step 0, then RL *amplifies* pre-existing behavior rather than creating it, and a
rising response length during training is not by itself evidence of 'emergence'
(the oat-zero / SimpleRL-Zoo observation).

Runs on CPU/MPS with no training. Usage:
    uv run python scripts/behavioral_step0.py --model Qwen/Qwen2.5-0.5B \
        --num-puzzles 64 --group-size 2 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from grpo.config import GenCfg, ModelCfg
from grpo.countdown import make_dataset, render_prompt, score_completion
from grpo.models import load_model_and_tokenizer, pick_device

# Cognitive-behavior lexicon. Deliberately conservative word-boundary patterns;
# counted only in the *completion* (what the model generated), never the prompt.
BEHAVIORS = {
    "verification": re.compile(
        r"\b(check|verif(?:y|ies|ied|ication)|confirm|make sure|double[ -]?check|"
        r"is this (?:right|correct)|is that (?:right|correct)|let me see if|"
        r"which (?:is|gives|equals))\b",
        re.I,
    ),
    "backtracking": re.compile(
        r"\b(wait|hmm+|actually|let me reconsider|reconsider|instead|"
        r"on second thought|scratch that|try again|that'?s (?:not|wrong|incorrect)|"
        r"that is (?:not|wrong)|doesn'?t (?:work|equal)|no,? that|but that)\b",
        re.I,
    ),
    "enumeration": re.compile(
        r"\b(let me try|another (?:way|approach|option|combination)|alternativ|"
        r"what if|we (?:could|can) (?:try|use)|option \d|"
        r"first,|then,|next,|or we)\b",
        re.I,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--num-puzzles", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch-prompts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/step0_behavior")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device("auto")
    print(f"device={device} model={args.model}", flush=True)

    model, tokenizer = load_model_and_tokenizer(
        ModelCfg(name=args.model, dtype="auto"), device
    )
    model.eval()
    gen = GenCfg(group_size=args.group_size, temperature=args.temperature, max_new_tokens=args.max_new_tokens)

    puzzles = make_dataset(n=args.num_puzzles, seed=args.seed)

    records: list[dict] = []
    t0 = time.time()
    for start in range(0, len(puzzles), args.batch_prompts):
        chunk = puzzles[start : start + args.batch_prompts]
        rendered = [render_prompt(p, "r1") for p in chunk]  # (text, think_open=True)
        prompts = [t for t, _ in rendered]
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
        comp_ids = out[:, enc["input_ids"].shape[1] :]
        comps = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
        comp_tok_len = (comp_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
        for i, p in enumerate(chunk):
            for g in range(args.group_size):
                text = comps[i * args.group_size + g]
                sc = score_completion(text, p, think_open=True, mode="shaped")
                hits = {k: bool(rx.search(text)) for k, rx in BEHAVIORS.items()}
                records.append(
                    {
                        "text": text,
                        "tok_len": int(comp_tok_len[i * args.group_size + g]),
                        "format_ok": sc.format_ok,
                        "answer_ok": sc.answer_ok,
                        **{f"beh_{k}": v for k, v in hits.items()},
                    }
                )
        print(f"  {len(records)} completions ({time.time() - t0:.0f}s)", flush=True)

    n = len(records)
    frac = lambda key: sum(r[key] for r in records) / n
    any_beh = sum(any(r[f"beh_{k}"] for k in BEHAVIORS) for r in records) / n
    summary = {
        "model": args.model,
        "n_completions": n,
        "mean_tok_len": sum(r["tok_len"] for r in records) / n,
        "format_ok_rate": frac("format_ok"),
        "answer_ok_rate": frac("answer_ok"),
        "behavior_any_rate": any_beh,
        **{f"behavior_{k}_rate": frac(f"beh_{k}") for k in BEHAVIORS},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    print("\n=== STEP-0 BEHAVIORAL ANALYSIS (base model, no RL) ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== example completions that show self-reflection ===")
    shown = 0
    for r in records:
        if any(r[f"beh_{k}"] for k in BEHAVIORS) and 20 < r["tok_len"] < 220:
            cats = [k for k in BEHAVIORS if r[f"beh_{k}"]]
            snippet = " ".join(r["text"].split())[:300]
            print(f"\n[{'/'.join(cats)}] {snippet}")
            shown += 1
            if shown >= 4:
                break


if __name__ == "__main__":
    main()
