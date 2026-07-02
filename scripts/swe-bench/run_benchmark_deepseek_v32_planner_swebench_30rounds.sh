#!/usr/bin/env bash
# Full SWE-Bench router-planning evaluation for deepseek-v3.2 with 30 Uno rounds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export UNO_MAX_ROUNDS=30
export SWE_MODE=planning
RUN_TS="$(date +%Y%m%d_%H%M)"
export OUT_SUFFIX="${OUT_SUFFIX:-30rounds_${RUN_TS}}"

MAX_TASKS_ARGS=(--max-tasks 10)
for arg in "$@"; do
  if [[ "${arg}" == "--max-tasks" ]]; then
    MAX_TASKS_ARGS=()
    break
  fi
done

bash "${SCRIPT_DIR}/run_benchmark_deepseek_v32_planner_swebench.sh" \
  --router-planning \
  "${MAX_TASKS_ARGS[@]}" \
  "$@"
