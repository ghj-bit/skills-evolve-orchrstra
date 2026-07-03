#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi

API_KEY="${DEEPSEEK_API_KEY:-${API_KEY:-}}"
API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
PLANNER_MODEL_ID="${PLANNER_MODEL_ID:-deepseek-v4-flash}"
ROUTER_MODEL_ID="${ROUTER_MODEL_ID:-${PLANNER_MODEL_ID}}"
WORKER_MODEL_ID="${WORKER_MODEL_ID:-deepseek-v4-pro}"
EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
RUN_NAME="${RUN_NAME:-rubric_workflow_2020b_uno_planner}"
TIMESTAMP="$(date +%Y%m%d_%H%M)"
OUT_DIR="${OUT_DIR:-${EVAL_OUT}/${RUN_NAME}_${TIMESTAMP}}"

if [[ -z "${API_KEY}" ]]; then
    echo "DEEPSEEK_API_KEY or API_KEY must be set before running this script." >&2
    exit 2
fi

for cert_var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
    cert_path="${!cert_var:-}"
    if [[ -n "${cert_path}" && ! -f "${cert_path}" ]]; then
        unset "${cert_var}"
    fi
done

export UNO_POOLS_PATH="${UNO_POOLS_PATH:-${PROJECT_DIR}/configs/pools.deepseek_v32.yaml}"
export RUBRIC_WORKFLOW_WORKER_MODELS="${RUBRIC_WORKFLOW_WORKER_MODELS:-${WORKER_MODEL_ID}}"
export RUBRIC_WORKFLOW_MAX_ATTEMPTS="${RUBRIC_WORKFLOW_MAX_ATTEMPTS:-8}"
export RUBRIC_WORKFLOW_SUBAGENT_MAX_STEPS="${RUBRIC_WORKFLOW_SUBAGENT_MAX_STEPS:-40}"
export RUBRIC_WORKFLOW_CMD_TIMEOUT="${RUBRIC_WORKFLOW_CMD_TIMEOUT:-7200}"
if [[ -z "${GIT_BASH:-}" && -n "${BASH:-}" ]]; then
    if command -v cygpath >/dev/null 2>&1; then
        export GIT_BASH="$(cygpath -w "${BASH}")"
    elif [[ -x "${BASH}" ]]; then
        export GIT_BASH="${BASH}"
    fi
fi

mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/run.log"
RUN_PID=""

cleanup() {
    local signal="${1:-INT}"
    echo "Received ${signal}; stopping eval process..." >> "${LOG_FILE}"
    if [[ -n "${RUN_PID}" ]]; then
        kill -INT "${RUN_PID}" 2>/dev/null || true
        sleep 5
        if kill -0 "${RUN_PID}" 2>/dev/null; then
            echo "Eval process still running; sending TERM..." >> "${LOG_FILE}"
            kill -TERM "${RUN_PID}" 2>/dev/null || true
            sleep 2
        fi
        if kill -0 "${RUN_PID}" 2>/dev/null; then
            echo "Eval process still running; sending KILL..." >> "${LOG_FILE}"
            kill -KILL "${RUN_PID}" 2>/dev/null || true
        fi
    fi
    exit 130
}

trap 'cleanup INT' INT
trap 'cleanup TERM' TERM

{
    echo "========================================"
    echo "RubricWorkflow Uno Planner Eval"
    echo "Project:       ${PROJECT_DIR}"
    echo "Fixture root:  ${PROJECT_DIR}/eval_pipeline/benchmarks/rubric_workflow_fixtures"
    echo "Scorer package:${RUBRIC_WORKFLOW_PACKAGE_ROOT:-installed/importable}"
    echo "Task IDs:      ${RUBRIC_WORKFLOW_TASK_IDS:-default-in-python}"
    echo "Planner model: ${PLANNER_MODEL_ID}"
    echo "Router model:  ${ROUTER_MODEL_ID}"
    echo "Worker model:  ${WORKER_MODEL_ID}"
    echo "API_BASE:      ${API_BASE}"
    echo "Output:        ${OUT_DIR}"
    echo "UNO_POOLS_PATH:${UNO_POOLS_PATH}"
    echo "GIT_BASH:      ${GIT_BASH:-}"
    echo "LANG:          ${LANG:-}"
    echo "LC_ALL:        ${LC_ALL:-}"
    echo "PYTHONUTF8:    ${PYTHONUTF8:-}"
    echo "PYTHONIOENCODING:${PYTHONIOENCODING:-}"
    echo "========================================"
} > "${LOG_FILE}"

cd "${PROJECT_DIR}"
python -u -m eval_pipeline \
    --router planner \
    --bench rubric_workflow \
    --api_key "${API_KEY}" \
    --api_base "${API_BASE}" \
    --local_base "${API_BASE}" \
    --local_model "${PLANNER_MODEL_ID}" \
    --router_model "${ROUTER_MODEL_ID}" \
    --output_dir "${OUT_DIR}" \
    --verify_workers 1 \
    --pass-k 1 \
    --interactive >> "${LOG_FILE}" 2>&1 &
RUN_PID=$!
wait "${RUN_PID}"
STATUS=$?
trap - INT TERM

if [[ "${STATUS}" -ne 0 ]]; then
    echo "Command failed with status ${STATUS}. Results: ${OUT_DIR}" >> "${LOG_FILE}"
    exit "${STATUS}"
fi

{
    echo ""
    echo "=== RubricWorkflow Score Summary ==="
    python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
trajectories = sorted((out_dir / "logs").glob("*/trajectory.json"))
if not trajectories:
    print("No trajectory.json found; score is unavailable.")
    raise SystemExit(0)

for trajectory_path in trajectories:
    try:
        data = json.loads(trajectory_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"{trajectory_path.parent.name}: failed to read trajectory: {exc}")
        continue

    task_id = data.get("task_id") or trajectory_path.parent.name
    score_info = data.get("score_info") or {}
    error = score_info.get("error") or data.get("last_error")
    raw_score = score_info.get("raw_score")
    full_score = score_info.get("full_score")
    reward = score_info.get("reward", data.get("reward", 0.0))

    if raw_score is not None and full_score:
        pct = float(raw_score) / float(full_score) * 100.0
        print(f"{task_id}: {float(raw_score):.2f}/{float(full_score):.2f} ({pct:.2f}%)")
    else:
        print(f"{task_id}: reward={float(reward or 0.0):.4f}")

    if score_info.get("score_json"):
        print(f"  score_json: {score_info['score_json']}")
    if score_info.get("report_path"):
        print(f"  report: {score_info['report_path']}")
    if error:
        print(f"  error: {error}")
PY
    echo "====================================="
    echo "Done. Results: ${OUT_DIR}"
} | tee -a "${LOG_FILE}"
