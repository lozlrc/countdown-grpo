# Runbook: GPU runs

Everything in this repo was built and tested locally for free (CPU/MPS).
This file is the exact procedure for the paid part: rented-GPU training
runs. Nothing here needs to be re-derived at rental time.

## 1. Provider and cost

Prices checked July 2026; re-check before renting.

| GPU | Where | $/hr | Use for |
| --- | --- | --- | --- |
| RTX 4090 24GB | RunPod Community | ~0.34 | `qwen15b_4090.yaml` |
| RTX 4090 24GB | RunPod Secure / Vast.ai | 0.59-0.69 | same, if community capacity is flaky |
| A100 80GB | Vast.ai | 0.67-0.79 | `qwen3b_a100.yaml` |
| A100 80GB | RunPod Secure | ~0.79 | same |

Per-run cost estimates (HF `generate` is the bottleneck; no vLLM here):

- 1.5B / 4090, 300 steps at ~60-90 s/step: 5-8 h, **$2-6 per run**
- 3B / A100, 500 steps at ~90-150 s/step: 12-20 h, **$10-16 per run**
- Full ablation matrix on 1.5B (see section 6): **$50-150 total project**

If step time makes the 3B run exceed budget, the highest-ROI optimization
is a vLLM rollout worker (generation is >70% of wall-clock); that is a
deliberate non-goal for the from-scratch version.

## 2. Box setup (Ubuntu CUDA image, e.g. runpod/pytorch)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
git clone <this-repo> && cd countdown-grpo
uv sync
uv run pytest -q          # must be green before spending GPU-hours
```

Optional wandb (recommended for curve comparison against TinyZero):

```bash
uv sync --extra wandb
uv run wandb login        # paste API key
# then set train.wandb: true in the config
```

Run inside tmux so an SSH drop does not kill the run:

```bash
tmux new -s grpo
uv run python -m grpo.train --config configs/qwen15b_4090.yaml
```

Resume after interruption:

```bash
uv run python -m grpo.train --config configs/qwen15b_4090.yaml \
    --resume runs/qwen15b_4090/ckpt_last.pt
```

Pull metrics off the box (JSONL, one line per step):

```bash
scp <box>:countdown-grpo/runs/qwen15b_4090/metrics.jsonl .
```

## 3. Run order

1. `qwen15b_4090.yaml` — first takeoff attempt. 1.5B is the smallest scale
   TinyZero reports working for Countdown zero-RL. Cheap enough to iterate.
2. `qwen3b_a100.yaml` — headline run, only after 1.5B shows a reward curve.
3. `qwen3b_4090_offload.yaml` — budget alternative for 3B. Read the memory
   note at the top of that config first: it needs an 8-bit optimizer
   (bitsandbytes) wired into `train.py` before it fits in 24GB.

Do NOT bother with 0.5B on the GPU: it fails Countdown zero-RL (TinyZero
finding) and it already served its purpose as the free local pipeline check.

## 4. Expected milestones

Reference points from philschmid's mini-R1 (3B, similar recipe) and the
TinyZero public curves at https://wandb.ai/jiayipan/TinyZero:

| Step | Expectation |
| --- | --- |
| ~50 | format reward learned (format_rate near 1.0, reward_mean ~0.1+) |
| ~100 | ~25% answer accuracy |
| ~200 | ~40% answer accuracy, completions lengthen as search behavior appears |

If format_rate is still ~0 at step 100 on 1.5B, something is wrong —
check a sampled transcript before burning more hours.

## 5. Failure modes and dials

- **NaN loss**: the loop raises immediately (by design). Halve `optim.lr`,
  resume from `ckpt_last.pt`.
- **Entropy collapse** (repeated tokens, `mean_action_logprob` drifting
  toward 0, all-identical completions): raise temperature to 1.2, or add
  the KL term (`kl_beta: 0.001` with `lr: 5.0e-7` — philschmid's stable
  setting; beta 0.04 from the original GRPO paper is documented unstable
  at these scales).
- **degenerate_frac near 1.0**: all G rewards identical within groups, so
  no learning signal. Early on this means the task is too hard (all zeros)
  — confirm format_rate is moving; late it means saturation.
- **completion_len exploding without accuracy gains**: classic GRPO length
  bias. Switch `loss.agg: drgrpo` and `loss.adv_use_std: false` (Dr. GRPO,
  arXiv 2503.20783).
- **OOM in the update**: halve `train.minibatch_size`, double
  `train.grad_accum` (same effective batch).

## 6. Ablation matrix

2 seeds each (`train.seed`), on 1.5B/4090 unless noted. Each cell is one
config override on top of `qwen15b_4090.yaml`.

| Ablation | Values | Knob |
| --- | --- | --- |
| Group size | G = 4 / 8 / 16 | `gen.group_size` |
| KL penalty | off / beta 0.001 (lr 5e-7) | `loss.kl_beta`, `optim.lr` |
| Reward shaping | shaped (0.1 format tier) / answer_only | `reward.mode` |
| Base vs instruct | Qwen2.5-1.5B / -1.5B-Instruct | `model.name` (+ `data.template: chat` for instruct) |
| Prompt template | r1 / plain | `data.template` |
| Dr. GRPO | std+token_mean / no-std+drgrpo | `loss.adv_use_std`, `loss.agg` |

Priority if budget-constrained: Dr. GRPO > KL > reward shaping > group
size > template > base-vs-instruct. Baseline (2 seeds) + 6 ablations
x 2 arms x 2 seeds at $2-6/run fits in the $50-150 envelope; cut seeds
to 1 for the cheap-to-interpret ablations (template, base-vs-instruct)
if needed.

## 7. Curve comparison

Plot `eval_answer_rate` and `format_rate` from `metrics.jsonl` against the
TinyZero public wandb (link above; their `critic/score/mean` is the reward
analog). Milestone table in section 4 is the pass/fail bar for the
replication claim.
