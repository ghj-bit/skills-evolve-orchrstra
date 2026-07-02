#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi

for cert_var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
    cert_path="${!cert_var:-}"
    if [[ -n "${cert_path}" && ! -f "${cert_path}" ]]; then
        unset "${cert_var}"
    fi
done

export UNO_POOLS_PATH="${UNO_POOLS_PATH:-${PROJECT_DIR}/configs/pools.deepseek_v32.yaml}"
export TERMINALBENCH_BUDGET_AS_USER="${TERMINALBENCH_BUDGET_AS_USER:-1}"
export DIRECT_ROUTER_DEBUG="${DIRECT_ROUTER_DEBUG:-1}"
export DIRECT_ROUTER_TIMEOUT="${DIRECT_ROUTER_TIMEOUT:-120}"
export DIRECT_ROUTER_MAX_RETRIES="${DIRECT_ROUTER_MAX_RETRIES:-1}"

EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
RUN_NAME="${RUN_NAME:-direct_deepseek_v4_flash_terminalbench}"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
OUT_DIR="${OUTPUT_DIR:-${EVAL_OUT}/${RUN_NAME}_${TIMESTAMP}}"
LOG_FILE="${OUT_DIR}/run.log"

MAX_STEPS="${MAX_STEPS:-30}"
CMD_TIMEOUT="${CMD_TIMEOUT:-300}"

mkdir -p "${OUT_DIR}"

CMD=(
    python -u scripts/run_terminalbench_direct_gpt_oss_120b_free.py
    --output-dir "${OUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --cmd-timeout "${CMD_TIMEOUT}"
)

if [[ -n "${MAX_TASKS:-}" ]]; then
    CMD+=(--max-tasks "${MAX_TASKS}")
fi

if [[ -n "${TASK_IDS:-}" ]]; then
    CMD+=(--task-ids "${TASK_IDS}")
fi

if [[ "${RESUME:-0}" != "0" ]]; then
    CMD+=(--resume)
fi

{
    echo "========================================"
    echo "Terminal-Bench Direct DeepSeek v4 Flash Eval"
    echo "Project:       ${PROJECT_DIR}"
    echo "Model:         deepseek-v4-flash"
    echo "UNO_POOLS_PATH:${UNO_POOLS_PATH}"
    echo "TERMINALBENCH_BUDGET_AS_USER:${TERMINALBENCH_BUDGET_AS_USER}"
    echo "DIRECT_ROUTER_DEBUG:${DIRECT_ROUTER_DEBUG}"
    echo "DIRECT_ROUTER_TIMEOUT:${DIRECT_ROUTER_TIMEOUT}"
    echo "DIRECT_ROUTER_MAX_RETRIES:${DIRECT_ROUTER_MAX_RETRIES}"
    echo "Output:        ${OUT_DIR}"
    echo "Max steps:     ${MAX_STEPS}"
    echo "Cmd timeout:   ${CMD_TIMEOUT}"
    echo "MAX_TASKS:     ${MAX_TASKS:-ALL}"
    echo "========================================"
} > "${LOG_FILE}"

cd "${PROJECT_DIR}"
"${CMD[@]}" >> "${LOG_FILE}" 2>&1

{
    echo "Done. Results: ${OUT_DIR}"
} >> "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}"
