"""Config-driven GRPO training loop.

Usage:
    uv run python -m grpo.train --config configs/debug_tiny_cpu.yaml
    uv run python -m grpo.train --config ... --resume runs/x/ckpt_last.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch

from grpo.advantage import group_advantages
from grpo.config import Config, load_config
from grpo.countdown import make_dataset, render_prompt, score_completion
from grpo.loss import grpo_loss
from grpo.models import load_model_and_tokenizer, pick_device
from grpo.rollout import (
    RolloutBatch,
    batched_token_logprobs,
    sample_rollouts,
    token_logprobs,
)


def grpo_update(
    model,
    optimizer,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    cfg: Config,
    ref_logprobs: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run epochs_per_batch passes of minibatched clipped policy updates.

    Fails loudly on non-finite loss rather than continuing a poisoned run.
    """
    model.train()
    n = batch.sequences.shape[0]
    accum = cfg.train.grad_accum
    mb_size = cfg.train.minibatch_size
    stats_acc: dict[str, float] = {}
    grad_norm = 0.0
    num_losses = 0

    for _ in range(cfg.train.epochs_per_batch):
        perm = torch.randperm(n)
        minibatches = [perm[i : i + mb_size] for i in range(0, n, mb_size)]
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for k, idx in enumerate(minibatches):
            logprobs = token_logprobs(
                model,
                batch.sequences[idx],
                batch.attention_mask[idx],
                temperature=cfg.gen.temperature,
            )
            loss, stats = grpo_loss(
                logprobs=logprobs,
                old_logprobs=batch.old_logprobs[idx],
                advantages=advantages[idx],
                action_mask=batch.action_mask[idx],
                clip_eps=cfg.loss.clip_eps,
                kl_beta=cfg.loss.kl_beta,
                ref_logprobs=ref_logprobs[idx] if ref_logprobs is not None else None,
                agg=cfg.loss.agg,
                max_tokens=cfg.gen.max_new_tokens,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {loss.item()} in policy update")
            (loss / accum).backward()
            pending += 1
            for key, val in stats.items():
                stats_acc[key] = stats_acc.get(key, 0.0) + val
            num_losses += 1
            if pending == accum or k == len(minibatches) - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optim.grad_clip
                ).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0

    out = {k: v / max(num_losses, 1) for k, v in stats_acc.items()}
    out["grad_norm"] = grad_norm
    return out


@torch.no_grad()
def evaluate(model, tokenizer, puzzles, cfg: Config, device) -> dict[str, float]:
    """Greedy decode on held-out puzzles; report answer/format rates."""
    model.eval()
    answer_hits = 0
    format_hits = 0
    bs = max(cfg.train.minibatch_size, 1)
    for i in range(0, len(puzzles), bs):
        chunk = puzzles[i : i + bs]
        rendered = [render_prompt(p, cfg.data.template, tokenizer) for p in chunk]
        prompts = [r[0] for r in rendered]
        think_open = rendered[0][1]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        seqs = model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            do_sample=False,
            max_new_tokens=cfg.train.eval_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = tokenizer.batch_decode(
            seqs[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for puzzle, completion in zip(chunk, completions):
            s = score_completion(
                completion,
                puzzle,
                think_open=think_open,
                mode=cfg.reward.mode,
                format_reward=cfg.reward.format_reward,
                answer_reward=cfg.reward.answer_reward,
            )
            answer_hits += s.answer_ok
            format_hits += s.format_ok
    return {
        "eval_answer_rate": answer_hits / len(puzzles),
        "eval_format_rate": format_hits / len(puzzles),
    }


def save_checkpoint(path: Path, model, optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def train(cfg: Config, resume: str | None = None) -> None:
    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    print(f"device: {device}")

    model, tokenizer = load_model_and_tokenizer(cfg.model, device)
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ref_model = None
    if cfg.loss.kl_beta > 0.0:
        ref_model = copy.deepcopy(model).eval()
        ref_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    start_step = 0
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        print(f"resumed from {resume} at step {start_step}")

    data_kwargs = dict(
        num_counts=cfg.data.num_counts,
        nums_min=cfg.data.nums_min,
        nums_max=cfg.data.nums_max,
        target_min=cfg.data.target_min,
        target_max=cfg.data.target_max,
    )
    eval_puzzles = make_dataset(cfg.data.eval_size, seed=cfg.data.eval_seed, **data_kwargs)
    eval_keys = {p.key for p in eval_puzzles}
    train_rng = random.Random(cfg.train.seed + start_step)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    wandb_run = None
    if cfg.train.wandb:
        import dataclasses

        import wandb

        wandb_run = wandb.init(
            project=cfg.train.wandb_project, config=dataclasses.asdict(cfg)
        )

    for step in range(start_step + 1, cfg.train.total_steps + 1):
        t0 = time.time()
        lr = cfg.optim.lr * min(1.0, step / max(cfg.optim.warmup_steps, 1))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # fresh puzzles every step, held-out set excluded
        puzzles = []
        while len(puzzles) < cfg.train.prompts_per_step:
            p = make_dataset(1, seed=train_rng.randrange(2**31), exclude=eval_keys, **data_kwargs)[0]
            puzzles.append(p)
        rendered = [render_prompt(p, cfg.data.template, tokenizer) for p in puzzles]
        prompts = [r[0] for r in rendered]
        think_open = rendered[0][1]

        batch = sample_rollouts(
            model,
            tokenizer,
            prompts,
            cfg.gen,
            device,
            logprob_minibatch=cfg.train.minibatch_size,
        )

        scores = [
            score_completion(
                completion,
                puzzles[i // cfg.gen.group_size],
                think_open=think_open,
                mode=cfg.reward.mode,
                format_reward=cfg.reward.format_reward,
                answer_reward=cfg.reward.answer_reward,
            )
            for i, completion in enumerate(batch.completions)
        ]
        rewards = torch.tensor([s.reward for s in scores], device=device)
        advantages, degenerate = group_advantages(
            rewards, cfg.gen.group_size, use_std=cfg.loss.adv_use_std
        )

        metrics = {
            "step": step,
            "reward_mean": rewards.mean().item(),
            "answer_rate": sum(s.answer_ok for s in scores) / len(scores),
            "format_rate": sum(s.format_ok for s in scores) / len(scores),
            "completion_len": batch.action_mask.sum(dim=1).mean().item(),
            "mean_action_logprob": (
                (batch.old_logprobs * batch.action_mask).sum()
                / batch.action_mask.sum().clamp(min=1.0)
            ).item(),
            "degenerate_frac": degenerate.float().mean().item(),
            "lr": lr,
        }

        keep = ~degenerate if cfg.loss.filter_degenerate else torch.ones_like(degenerate)
        if keep.any():
            if cfg.loss.filter_degenerate and not keep.all():
                idx = keep.nonzero(as_tuple=True)[0]
                sub = RolloutBatch(
                    sequences=batch.sequences[idx],
                    attention_mask=batch.attention_mask[idx],
                    action_mask=batch.action_mask[idx],
                    old_logprobs=batch.old_logprobs[idx],
                    completions=[batch.completions[i] for i in idx.tolist()],
                    prompt_len=batch.prompt_len,
                    group_size=batch.group_size,
                )
                sub_adv = advantages[idx]
            else:
                sub, sub_adv = batch, advantages
            ref_logprobs = None
            if ref_model is not None:
                ref_logprobs = batched_token_logprobs(
                    ref_model,
                    sub.sequences,
                    sub.attention_mask,
                    cfg.train.minibatch_size,
                    temperature=cfg.gen.temperature,
                )
            metrics.update(grpo_update(model, optimizer, sub, sub_adv, cfg, ref_logprobs))
        else:
            metrics["skipped"] = 1.0  # every group had identical rewards

        metrics["step_time"] = time.time() - t0

        if step % cfg.train.eval_interval == 0:
            metrics.update(evaluate(model, tokenizer, eval_puzzles, cfg, device))
            print(f"[eval] step {step}: {metrics['eval_answer_rate']:.3f} answer, "
                  f"{metrics['eval_format_rate']:.3f} format")

        if step % cfg.train.log_interval == 0:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            print(
                f"step {step}: reward {metrics['reward_mean']:.3f} "
                f"answer {metrics['answer_rate']:.3f} format {metrics['format_rate']:.3f} "
                f"len {metrics['completion_len']:.0f} ({metrics['step_time']:.1f}s)"
            )

        if step % cfg.train.ckpt_interval == 0:
            save_checkpoint(out_dir / "ckpt_last.pt", model, optimizer, step)

    save_checkpoint(out_dir / "ckpt_final.pt", model, optimizer, cfg.train.total_steps)
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training on Countdown")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
