#!/usr/bin/env bash
# Full SWE-Bench router-planning evaluation for deepseek-v3.2 with 8 Uno rounds.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

export UNO_MAX_ROUNDS=8
export SWE_MODE=planning
export OUT_SUFFIX=8rounds
export MODEL_ID="${MODEL_ID:-deepseek-v4-flash}"
export DEEPSEEK_BASE="${DEEPSEEK_BASE:-https://api.deepseek.com}"
export LOCAL_BASE="${LOCAL_BASE:-${DEEPSEEK_BASE}}"
export API_BASE="${API_BASE:-${DEEPSEEK_BASE}}"
export API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-}}"

if [[ -z "${API_KEY}" ]]; then
  echo "API_KEY or DEEPSEEK_API_KEY must be set before running this script." >&2
  exit 2
fi

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
