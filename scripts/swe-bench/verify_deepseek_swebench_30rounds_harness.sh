#!/usr/bin/env bash
# Verify existing DeepSeek 30-round SWE-Bench predictions with the official harness.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MAX_WORKERS="${MAX_WORKERS:-2}"
SWEBENCH_CONDA_ENV="${SWEBENCH_CONDA_ENV:-swebench}"
EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
MODEL_NAME="${MODEL_NAME:-deepseek}"
EVAL_ROOT="${EVAL_OUT}/${MODEL_NAME}_planner"
PREDICTIONS="${PREDICTIONS:-${EVAL_ROOT}/swebench_router_planning_30rounds/predictions.jsonl}"
WORK_DIR="${WORK_DIR:-${EVAL_ROOT}/swebench_router_planning_30rounds/harness_network}"

python scripts/verify_deepseek_swebench_30rounds_harness.py \
  --predictions "${PREDICTIONS}" \
  --max-workers "${MAX_WORKERS}" \
  --conda-env "${SWEBENCH_CONDA_ENV}" \
  --work-dir "${WORK_DIR}" \
  "$@"
