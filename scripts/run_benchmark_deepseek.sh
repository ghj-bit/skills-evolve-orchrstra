#!/usr/bin/env bash
# Direct benchmark runner for deepseek.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_NAME="deepseek"
MODEL_ID="${MODEL_ID:-deepseek}"
LOCAL_BASE="${LOCAL_BASE:-https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/4ff709dd-915e-4392-8a69-12c61dc95edb/ae01f99f-3457-4c25-a7ce-ed967ad2ff02/proxy/8055/v1}"
API_KEY="${API_KEY:-empty}"
API_BASE="${API_BASE:-${LOCAL_BASE}}"
EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
GEN_WORKERS="${GEN_WORKERS:-4}"
VERIFY_WORKERS="${VERIFY_WORKERS:-2}"
PASS_K="${PASS_K:-1}"

ALL_BENCHMARKS=(terminalbench)
SELECTED_BENCHMARKS=()
MAX_TASKS_ARG=()
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench)
      IFS=',' read -ra SELECTED_BENCHMARKS <<< "$2"
      shift 2
      ;;
    --max-tasks)
      MAX_TASKS_ARG=(--max_tasks "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ ${#SELECTED_BENCHMARKS[@]} -eq 0 ]]; then
  SELECTED_BENCHMARKS=("${ALL_BENCHMARKS[@]}")
fi

echo "Model: ${MODEL_NAME}"
echo "Endpoint: ${LOCAL_BASE}"
echo "Benchmarks: ${SELECTED_BENCHMARKS[*]}"
echo "Output root: ${EVAL_OUT}/${MODEL_NAME}"

for BENCH in "${SELECTED_BENCHMARKS[@]}"; do
  OUT_DIR="${EVAL_OUT}/${MODEL_NAME}/${BENCH}"
  BENCH_EXTRA=()
  if [[ "${BENCH}" == "swebench" || "${BENCH}" == "terminalbench" ]]; then
    BENCH_EXTRA=(--interactive)
  fi

  CMD=(
    python -m eval_pipeline
    --router direct
    --bench "${BENCH}"
    --api_key "${API_KEY}"
    --api_base "${API_BASE}"
    --local_base "${LOCAL_BASE}"
    --local_model "${MODEL_ID}"
    --output_dir "${OUT_DIR}"
    --gen_workers "${GEN_WORKERS}"
    --verify_workers "${VERIFY_WORKERS}"
    --pass-k "${PASS_K}"
    "${BENCH_EXTRA[@]}"
    "${MAX_TASKS_ARG[@]}"
  )

  if [[ -f "${OUT_DIR}/summary.json" ]]; then
    echo "[SKIP] ${MODEL_NAME} ${BENCH}"
    continue
  fi
  mkdir -p "$(dirname "${OUT_DIR}")"

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[DRY] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
  else
    echo "[RUN] ${MODEL_NAME} ${BENCH}"
    "${CMD[@]}" 2>&1 | tee "${OUT_DIR}.log"
  fi
done

python scripts/collect_results.py --root "${EVAL_OUT}" --format md
