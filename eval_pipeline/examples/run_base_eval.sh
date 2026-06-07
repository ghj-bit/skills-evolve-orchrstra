#!/usr/bin/env bash
# Pre-SFT instruct baseline eval (GPUs 5,6,7)
# Usage:
#   bash eval_pipeline/examples/run_base_eval.sh            # launch
#   bash eval_pipeline/examples/run_base_eval.sh --status   # check progress
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
OC="${OPENCOMPASS_ROOT:/set OPENCOMPASS_ROOT to your OpenCompass checkout}"
CFG="${OPENCOMPASS_CONFIG:-opencompass/configs/eval_marl_base.py}"
OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval/base_opencompass}"
LOG="${OUT}.log"

if [[ "${1:-}" == "--status" ]]; then
    echo "=== Running processes ==="
    ps aux | grep "eval_marl_base" | grep -v grep || echo "(none)"
    echo "=== Latest results ==="
    latest=$(ls -td "$OUT"/2026* 2>/dev/null | head -1)
    [[ -n "$latest" ]] && find "$latest" -name "summary*" -exec cat {} \; 2>/dev/null || echo "(no results yet)"
    echo "=== Inference progress ==="
    [[ -n "$latest" ]] && find "$latest/predictions" -name "*.json" 2>/dev/null \
        | wc -l | xargs -I{} echo "{} prediction files done"
    exit 0
fi

mkdir -p "$OUT"
cd "$OC"

echo "Launching: $CFG -> $OUT"
PYTHONPATH="$OC:${PYTHONPATH:-}" \
nohup conda run -n marl python run.py "$CFG" --work-dir "$OUT" > "$LOG" 2>&1 &
echo "PID: $!  Log: $LOG"
