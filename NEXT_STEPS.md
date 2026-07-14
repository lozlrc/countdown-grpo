# Next steps — Countdown GRPO (R1-Zero replication)

Execution roadmap for the GPU phase. Rental mechanics, pricing, and the ablation
matrix live in `RUNBOOK.md`; this file is the ordered experiment plan and the
decision points, so you can run it without re-deriving anything.

**Current state:** trainer complete, 74 tests green, GRPO math audited against
DeepSeekMath eq. 3–4 (one real off-policy bug found & fixed), e2e policy-improvement
test passes (P(rewarded token) 0.01→0.235 in 30 steps), real-model MPS smoke ran on
Qwen2.5-0.5B. Everything below the GATE spends money — nothing before it does.

---

## Phase 0 — Free prep (no GPU, do anytime)
- [ ] `git remote add origin …` + push (repo is local-only; history clean, no attribution)
- [ ] Re-run `uv run pytest` locally to confirm green before you travel to a rented box
- [ ] Skim `RUNBOOK.md` cost table so you know the spend before committing

## Phase 1 — GPU box setup (GATE — the first thing that costs money)
- [ ] Rent a single **RTX 4090** (RunPod ~$0.34–0.69/hr or Vast ~$0.29–0.50/hr)
- [ ] Clone repo, `uv sync`, then **`uv run pytest` ON THE BOX before any training**
      — the GPU config batch/memory numbers are estimates flagged TODO-verify; a green
      suite on CUDA is the go/no-go signal
- [ ] `python scripts/smoke_qwen.py` with a CUDA config → confirm generate throughput
      and update RUNBOOK step-time estimates with the real number
- **Decision:** if pytest or the smoke fails on CUDA, fix before spending on training.

## Phase 2 — 0.5B debug run (cheap pipeline sanity, ~$1)
- [ ] Run a short 0.5B run on GPU to confirm the loop trains end-to-end on real weights
- **Expected:** format reward may rise slightly; accuracy stays low. This is the
  documented negative-result run (0.5B fails Countdown zero-RL per TinyZero) — keep the
  curve, it's a resume point, not a failure.

## Phase 3 — 1.5B first takeoff (the money curve, ~$2–6)
- [ ] `configs/qwen15b_4090.yaml`, base model (NOT instruct)
- [ ] Log to wandb (or the JSONL metrics) — watch for the R1-Zero signature:
      **format reward learned ~step 50, accuracy climbing ~step 100+, response length growing**
- **Milestone:** this is the headline figure. Compare your curve against TinyZero's
  public wandb (link in RUNBOOK). If reward is flat, first suspects: KL β too high
  (drop 0.04→0.001), prompt template mismatch, learning rate.

## Phase 4 — 3B headline run (~$10–16)
- [ ] `configs/qwen3b_a100.yaml` on an A100-80GB (or the 4090-offload config — note it
      needs an 8-bit optimizer wired in first; A100 is the clean path)
- [ ] Target the philschmid milestones: ~25% acc @ step 100, ~40% @ step 200

## Phase 5 — Ablation matrix (the differentiator — see RUNBOOK for the grid)
Each is a small run; this is what separates a real replication study from a fork.
2 seeds each where budget allows:
- [ ] **Group size G** sweep (e.g. 4 / 8 / 16) — stability & accuracy vs cost
- [ ] **KL on/off + β** (0 / 0.001 / 0.04) — reproduce the instability
- [ ] **Format-only vs format+answer reward** — decomposition
- [ ] **Base vs instruct** — format-learning speed
- [ ] **Prompt template** (plain / chat / r1) — the template-sensitivity result
- [ ] **Dr.GRPO on/off** — the length-bias correction (plot length-of-incorrect-responses)

## Phase 6 — Behavioral analysis (the "don't over-claim emergence" section)
- [ ] Count verification/backtracking/enumeration phrases at **step 0 vs end of training**
- [ ] Show self-reflection already exists in the base model (oat-zero finding) — this
      honesty is a maturity signal, not a weakness

## Phase 7 — Write up + publish
- [ ] Plots: reward curve, response-length curve, ablation bars, behavioral counts
- [ ] README: curves, the cost table ("$__ on one consumer GPU"), references, the
      explicit no-emergence framing
- [ ] Optional: short blog post — this is the kind of repo that gets shared

---

## Budget summary (full detail in RUNBOOK.md)
| Phase | Model | GPU | Est. cost |
|---|---|---|---|
| 1 setup + smoke | — | 4090 | ~$1 |
| 2 debug | 0.5B | 4090 | ~$1 |
| 3 takeoff | 1.5B | 4090 | ~$2–6 |
| 4 headline | 3B | A100-80GB | ~$10–16 |
| 5 ablations | 1.5B mostly | 4090 | ~$20–60 (grid × 2 seeds) |
| **Total** | | | **~$50–150** |

## Definition of done
A public repo with your own from-scratch GRPO trainer, a reward curve matching
TinyZero's shape, an ablation matrix with published precedents, a step-0 behavioral
analysis, and a "$__ on one GPU" cost line — plus the story of the off-policy
temperature bug you found and fixed.

## Resume bullet (fill after Phase 4)
> Implemented Group Relative Policy Optimization from scratch (pure PyTorch, no RL
> library) and reproduced DeepSeek R1-Zero-style reasoning emergence on the Countdown
> task with Qwen2.5-**_._**B; reward took off at step **__** for **$__** of compute on
> a single GPU. Ablated group size, KL penalty, and reward decomposition; validated
> the training loop with a policy-improvement unit test.

## Watch-outs (from the adversarial review)
- Rollout `temperature` is now threaded through all logprob paths — **`top_p` must stay
  1.0** (nucleus truncation has no logprob correction; the code raises if you change it).
- Advantage is 0 for zero-variance groups (all-identical rewards) — filtered, by design.
- Fail-loud on NaN is intentional; if you hit it, suspect KL β or LR before touching the loss.
