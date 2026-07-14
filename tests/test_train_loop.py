"""Full training-loop smoke tests on the tiny random model (CPU, fast)."""

from pathlib import Path

from grpo.config import load_config
from grpo.train import train

CONFIG = Path(__file__).parent.parent / "configs" / "debug_tiny_cpu.yaml"


def small_cfg(tmp_path, steps: int):
    cfg = load_config(CONFIG)
    cfg.train.total_steps = steps
    cfg.train.out_dir = str(tmp_path / "run")
    cfg.train.eval_interval = steps
    cfg.train.ckpt_interval = steps
    cfg.train.eval_max_new_tokens = 16
    cfg.gen.max_new_tokens = 24
    cfg.data.eval_size = 4
    return cfg


def test_debug_config_runs_and_logs(tmp_path):
    cfg = small_cfg(tmp_path, steps=2)
    train(cfg)
    out = Path(cfg.train.out_dir)
    metrics = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics) == 2
    assert (out / "ckpt_last.pt").exists()
    assert (out / "ckpt_final.pt").exists()


def test_resume_from_checkpoint(tmp_path):
    cfg = small_cfg(tmp_path, steps=2)
    train(cfg)
    ckpt = Path(cfg.train.out_dir) / "ckpt_last.pt"
    cfg.train.total_steps = 3
    train(cfg, resume=str(ckpt))
    metrics = (Path(cfg.train.out_dir) / "metrics.jsonl").read_text().strip().splitlines()
    # 2 lines from the first run, 1 more from the resumed step
    assert len(metrics) == 3
