#!/usr/bin/env bash
# One-shot setup + launch for a rented CUDA pod (RunPod-style PyTorch image).
#
# Usage (on the pod):
#   bash scripts/pod_setup.sh configs/qwen3b_a100.yaml
#
# Encodes the lessons from the first rented-GPU session:
#   - The image's SYSTEM python ships a torch matched to the pod's driver;
#     a uv-managed venv resolves a newer torch wheel that may NOT match
#     (CUDA-version mismatch / mixed nvidia-* stacks -> broken imports).
#     So: install the few pure-python deps into system python and run with
#     PYTHONPATH=src. No venv on the pod.
#   - PEP 668 marks system python "externally managed": --break-system-packages
#     is the standard container override.
#   - Always run the test suite on the box BEFORE burning GPU-hours.
#   - Train inside tmux (survives SSH drops); metrics stream to
#     runs/<name>/metrics.jsonl (flushed every step — tail that, not stdout,
#     which is pipe-buffered).
set -euo pipefail

CONFIG="${1:?usage: bash scripts/pod_setup.sh <config.yaml>}"
cd "$(dirname "$0")/.."

echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== system torch (should be CUDA-matched by the image) ==="
python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in system torch — wrong image?"
print(torch.__version__, "|", torch.cuda.get_device_name(0))
EOF

echo "=== deps into system python ==="
pip install -q --break-system-packages transformers tokenizers pyyaml numpy pytest

echo "=== test gate (must be green before spending GPU-hours) ==="
PYTHONPATH=src python3 -m pytest -q tests

echo "=== launch in tmux ==="
NAME=$(basename "$CONFIG" .yaml)
tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new-session -d -s "$NAME" \
  "HF_HOME=\${HF_HOME:-/workspace/hf} PYTHONPATH=src python3 -m grpo.train --config $CONFIG 2>&1 | tee train_$NAME.log; echo done > RUN_FINISHED_$NAME"
echo "launched tmux session '$NAME'"
echo "watch:   tail -f runs/$NAME/metrics.jsonl"
echo "attach:  tmux attach -t $NAME"
