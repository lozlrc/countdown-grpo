"""Config dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelCfg:
    name: str = "tiny-random"
    dtype: str = "auto"  # auto | float32 | bfloat16 | float16
    attn_implementation: str = "eager"  # sdpa/flash_attention_2 on real GPUs
    gradient_checkpointing: bool = False
    init_seed: int | None = 0  # tiny-random only


@dataclass
class DataCfg:
    num_counts: tuple[int, ...] = (3, 4)
    nums_min: int = 1
    nums_max: int = 99
    target_min: int = 10
    target_max: int = 99
    template: str = "r1"  # plain | chat | r1 (template choice is an ablation)
    eval_size: int = 128
    eval_seed: int = 7


@dataclass
class GenCfg:
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512


@dataclass
class RewardCfg:
    mode: str = "shaped"  # shaped | answer_only (ablation)
    format_reward: float = 0.1
    answer_reward: float = 1.0


@dataclass
class LossCfg:
    clip_eps: float = 0.2
    kl_beta: float = 0.0  # 0 disables KL and the reference model entirely
    agg: str = "token_mean"  # token_mean | seq_mean | drgrpo
    adv_use_std: bool = True  # False = Dr. GRPO advantage
    filter_degenerate: bool = True  # drop zero-variance groups from the update


@dataclass
class OptimCfg:
    lr: float = 1.0e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 10


@dataclass
class TrainCfg:
    total_steps: int = 200
    prompts_per_step: int = 8
    epochs_per_batch: int = 1
    minibatch_size: int = 8  # sequences per forward/backward
    grad_accum: int = 1  # minibatches per optimizer step
    eval_interval: int = 20
    eval_max_new_tokens: int = 512
    ckpt_interval: int = 50
    log_interval: int = 1
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs/default"
    wandb: bool = False
    wandb_project: str = "countdown-grpo"


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _build(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(raw)
    if "num_counts" in kwargs and kwargs["num_counts"] is not None:
        kwargs["num_counts"] = tuple(kwargs["num_counts"])
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return Config(
        model=_build(ModelCfg, raw.get("model")),
        data=_build(DataCfg, raw.get("data")),
        gen=_build(GenCfg, raw.get("gen")),
        reward=_build(RewardCfg, raw.get("reward")),
        loss=_build(LossCfg, raw.get("loss")),
        optim=_build(OptimCfg, raw.get("optim")),
        train=_build(TrainCfg, raw.get("train")),
    )
