#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml> [gpu_id]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="$1"
GPU_ID="${2:-0}"
PY_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi
if [ ! -x "$PY_BIN" ]; then
  echo "Python binary not found/executable: $PY_BIN" >&2
  exit 1
fi

CFG_BASENAME="$(basename "$CONFIG_PATH" .yaml)"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/runs/release/${CFG_BASENAME}_${TS}"
mkdir -p "$RUN_DIR"

cp "$CONFIG_PATH" "$RUN_DIR/config.yaml"
(git -C "$ROOT_DIR" rev-parse HEAD || true) > "$RUN_DIR/git_head.txt"
(git -C "$ROOT_DIR" status --short || true) > "$RUN_DIR/git_status.txt"

CMD=("$PY_BIN" "$ROOT_DIR/train.py" --config_file "$CONFIG_PATH")
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.txt"
echo >> "$RUN_DIR/command.txt"

echo "[run-release] run_dir=$RUN_DIR"
echo "[run-release] gpu=$GPU_ID"
echo "[run-release] config=$CONFIG_PATH"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"

if command -v rg >/dev/null 2>&1; then
  rg "Validation Results - Avg Loss" "$RUN_DIR/train.log" | tail -n 1 > "$RUN_DIR/final_validation_line.txt" || true
else
  grep "Validation Results - Avg Loss" "$RUN_DIR/train.log" | tail -n 1 > "$RUN_DIR/final_validation_line.txt" || true
fi

echo "[run-release] evaluating best checkpoint"
EXP_NAME="$($PY_BIN - <<'PY' "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('name', 'UNKNOWN'))
PY
)"

FINAL_EVAL_FLIP_TTA="$($PY_BIN - <<'PY' "$CONFIG_PATH"
import sys, yaml
with open(sys.argv[1], 'r') as f:
    cfg = yaml.safe_load(f)
print("true" if bool(cfg.get('final_eval_flip_tta', True)) else "false")
PY
)"
echo "[run-release] final_eval_flip_tta=$FINAL_EVAL_FLIP_TTA"

BEST_CKPT="$ROOT_DIR/checkpoints/$EXP_NAME/best_checkpoint.pt"
LAST_CKPT="$ROOT_DIR/checkpoints/$EXP_NAME/last_checkpoint.pt"
if [ -f "$BEST_CKPT" ]; then
  "$PY_BIN" "$ROOT_DIR/scripts/eval_spaceclip_checkpoint.py" --config "$CONFIG_PATH" --checkpoint "$BEST_CKPT" --flip-tta "$FINAL_EVAL_FLIP_TTA" --out "$RUN_DIR/best_metrics.json" \
    | tee "$RUN_DIR/best_metrics_stdout.json"
else
  echo "[run-release] best checkpoint not found: $BEST_CKPT" | tee "$RUN_DIR/eval_error.txt"
fi

echo "[run-release] evaluating last checkpoint"
if [ -f "$LAST_CKPT" ]; then
  "$PY_BIN" "$ROOT_DIR/scripts/eval_spaceclip_checkpoint.py" --config "$CONFIG_PATH" --checkpoint "$LAST_CKPT" --flip-tta "$FINAL_EVAL_FLIP_TTA" --out "$RUN_DIR/last_metrics.json" \
    | tee "$RUN_DIR/last_metrics_stdout.json"
else
  echo "[run-release] last checkpoint not found: $LAST_CKPT" | tee -a "$RUN_DIR/eval_error.txt"
fi

echo "[run-release] completed"
