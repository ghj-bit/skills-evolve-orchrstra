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
export TERMINALBENCH_DOCKER_MONITOR="${TERMINALBENCH_DOCKER_MONITOR:-1}"
export TERMINALBENCH_DOCKER_MONITOR_INTERVAL="${TERMINALBENCH_DOCKER_MONITOR_INTERVAL:-20}"
export TERMINALBENCH_DOCKER_IMAGE_PREFIX="${TERMINALBENCH_DOCKER_IMAGE_PREFIX:-alexgshaw}"
export TERMINALBENCH_DOCKER_IMAGE_TAG="${TERMINALBENCH_DOCKER_IMAGE_TAG:-20251031}"

EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
RUN_NAME="${RUN_NAME:-direct_deepseek_v4_flash_terminalbench}"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
OUT_DIR="${OUTPUT_DIR:-${EVAL_OUT}/${RUN_NAME}_${TIMESTAMP}}"
LOG_FILE="${OUT_DIR}/run.log"
RUN_PID=""

# shellcheck source=scripts/terminalbench_docker_cleanup_monitor.sh
source "${PROJECT_DIR}/scripts/terminalbench_docker_cleanup_monitor.sh"

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
    echo "docker_monitor:${TERMINALBENCH_DOCKER_MONITOR}"
    echo "docker_monitor_interval:${TERMINALBENCH_DOCKER_MONITOR_INTERVAL}"
    echo "docker_image_prefix:${TERMINALBENCH_DOCKER_IMAGE_PREFIX}"
    echo "docker_image_tag:${TERMINALBENCH_DOCKER_IMAGE_TAG}"
    echo "Output:        ${OUT_DIR}"
    echo "Max steps:     ${MAX_STEPS}"
    echo "Cmd timeout:   ${CMD_TIMEOUT}"
    echo "MAX_TASKS:     ${MAX_TASKS:-ALL}"
    echo "========================================"
} > "${LOG_FILE}"

cleanup() {
    local signal="${1:-INT}"
    echo "Received ${signal}; stopping eval process..." >> "${LOG_FILE}"
    terminalbench_stop_docker_cleanup_monitor
    if [[ -n "${RUN_PID}" ]]; then
        kill -INT "${RUN_PID}" 2>/dev/null || true
        sleep 5
        kill -TERM "${RUN_PID}" 2>/dev/null || true
    fi
    exit 130
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

cd "${PROJECT_DIR}"
terminalbench_start_docker_cleanup_monitor "${OUT_DIR}" "${LOG_FILE}"
"${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
STATUS=$?
trap - INT TERM
terminalbench_stop_docker_cleanup_monitor

if [[ "${STATUS}" -ne 0 ]]; then
    echo "Command failed with status ${STATUS}. Results: ${OUT_DIR}" >> "${LOG_FILE}"
    exit "${STATUS}"
fi

{
    echo "Done. Results: ${OUT_DIR}"
} >> "${LOG_FILE}"

echo "Done. Log: ${LOG_FILE}"
