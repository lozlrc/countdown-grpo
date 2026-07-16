"""Plot a training run's metrics.jsonl into the reward-takeoff curve.

Reads the per-step JSONL that train.py writes and renders four panels --
reward, answer accuracy, format rate, and mean completion length -- against
step, overlaying the sparse held-out eval points. This is the figure that
goes in the README once a GPU run finishes.

    uv run --with matplotlib python scripts/plot_metrics.py \
        runs/qwen15b_4090/metrics.jsonl --out docs/reward_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r and r.get(key) is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="path to runs/<name>/metrics.jsonl")
    ap.add_argument("--out", default="docs/reward_curve.png")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(Path(args.metrics))
    if not rows:
        raise SystemExit(f"no rows in {args.metrics}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("reward_mean", "train reward (group mean)", axes[0][0], None),
        ("answer_rate", "train answer accuracy", axes[0][1], "eval_answer_rate"),
        ("format_rate", "train format rate", axes[1][0], "eval_format_rate"),
        ("completion_len", "mean completion length (tokens)", axes[1][1], None),
    ]
    for key, label, ax, eval_key in panels:
        xs, ys = series(rows, key)
        ax.plot(xs, ys, lw=1.4, color="#3b6fe0", label="train")
        if eval_key:
            ex, ey = series(rows, eval_key)
            if ex:
                ax.plot(ex, ey, "o-", ms=4, lw=1.4, color="#e07b3b", label="held-out eval")
                ax.legend(fontsize=8, frameon=False)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("GRPO step")

    final = rows[-1]
    title = args.title or f"countdown-grpo — {Path(args.metrics).parent.name}"
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  (steps {rows[0]['step']}-{final['step']})")

    # a one-line text summary for the README/commit message
    def last(key: str):
        xs, ys = series(rows, key)
        return ys[-1] if ys else None

    print(
        "final: "
        + ", ".join(
            f"{k}={v:.3f}"
            for k, v in {
                "reward": last("reward_mean"),
                "eval_answer": last("eval_answer_rate"),
                "eval_format": last("eval_format_rate"),
                "len": last("completion_len"),
            }.items()
            if v is not None
        )
    )


if __name__ == "__main__":
    main()
