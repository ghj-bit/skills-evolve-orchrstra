#!/usr/bin/env bash
# Run a ToolBench/ToolACE evaluation through the public eval pipeline.
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

ROUTER="${ROUTER:-planner}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LOCAL_BASE="${LOCAL_BASE:-http://localhost:8000/v1}"
API_BASE="${REMOTE_API_BASE:-${API_BASE:-http://localhost:9000/v1}}"
API_KEY="${REMOTE_API_KEY:-${API_KEY:-EMPTY}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/eval/toolace_smoke}"
MAX_TASKS="${MAX_TASKS:-20}"
PASS_K="${PASS_K:-1}"
GEN_WORKERS="${GEN_WORKERS:-4}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"

mkdir -p "${OUT_DIR}"

echo "=== ToolBench / ToolACE evaluation ==="
echo "Router: ${ROUTER}"
echo "Local policy: ${LOCAL_MODEL} @ ${LOCAL_BASE}"
echo "Worker gateway: ${API_BASE}"
echo "Output: ${OUT_DIR}"
echo ""

"${PYTHON}" -m eval_pipeline \
  --router "${ROUTER}" \
  --bench toolbench \
  --api_key "${API_KEY}" \
  --api_base "${API_BASE}" \
  --local_base "${LOCAL_BASE}" \
  --local_model "${LOCAL_MODEL}" \
  --output_dir "${OUT_DIR}" \
  --max_tasks "${MAX_TASKS}" \
  --pass-k "${PASS_K}" \
  --gen_workers "${GEN_WORKERS}" \
  --verify_workers "${VERIFY_WORKERS}"

"${PYTHON}" scripts/collect_results.py --root "${REPO_ROOT}/data/eval" --format md
