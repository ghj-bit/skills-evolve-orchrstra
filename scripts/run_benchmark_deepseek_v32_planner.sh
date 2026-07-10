#!/usr/bin/env bash
# Planner benchmark runner for deepseek-v3.2.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_NAME="deepseek"
MODEL_ID="${MODEL_ID:-deepseek}"
LOCAL_BASE="${LOCAL_BASE:-https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/4ff709dd-915e-4392-8a69-12c61dc95edb/ae01f99f-3457-4c25-a7ce-ed967ad2ff02/proxy/8055/v1}"
API_KEY="${API_KEY:-empty}"
API_BASE="${API_BASE:-${LOCAL_BASE}}"
UNO_POOLS_PATH="${UNO_POOLS_PATH:-${PROJECT_DIR}/configs/pools.deepseek_v32.yaml}"
UNO_SYSTEM_PROMPT="${UNO_SYSTEM_PROMPT:-${PROJECT_DIR}/configs/uno/system_prompt_deepseek_v32.txt}"
export UNO_POOLS_PATH UNO_SYSTEM_PROMPT
EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_DIR}/data/huggingface/datasets}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_DIR}/data/huggingface/hub}"
GAIA_DATA_DIR="${GAIA_DATA_DIR:-${HF_HUB_CACHE}/datasets--gaia-benchmark--GAIA}"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE GAIA_DATA_DIR
GEN_WORKERS="${GEN_WORKERS:-4}"
VERIFY_WORKERS="${VERIFY_WORKERS:-2}"
PASS_K="${PASS_K:-1}"
UNO_VERBOSE_RESPONSES="${UNO_VERBOSE_RESPONSES:-1}"
UNO_MAX_ROUNDS="${UNO_MAX_ROUNDS:-8}"
export UNO_VERBOSE_RESPONSES UNO_MAX_ROUNDS

for cert_var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
  cert_path="${!cert_var:-}"
  if [[ -n "${cert_path}" && ! -f "${cert_path}" ]]; then
    echo "Unset ${cert_var}: file not found (${cert_path})"
    unset "${cert_var}"
  fi
done

ALL_BENCHMARKS=(gaia)
SELECTED_BENCHMARKS=()
MAX_TASKS_ARG=(--max_tasks 10)
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
echo "Output root: ${EVAL_OUT}/${MODEL_NAME}_planner"
echo "Uno pool: ${UNO_POOLS_PATH}"
echo "Uno prompt: ${UNO_SYSTEM_PROMPT}"
echo "HF cache: ${HF_HOME}"
echo "GAIA local data: ${GAIA_DATA_DIR}"

for BENCH in "${SELECTED_BENCHMARKS[@]}"; do
  OUT_DIR="${EVAL_OUT}/${MODEL_NAME}_planner/${BENCH}"
  BENCH_EXTRA=()
  RUN_ENV=()
  if [[ "${BENCH}" == "terminalbench" ]]; then
    BENCH_EXTRA=(--interactive)
  elif [[ "${BENCH}" == "swebench" ]]; then
    RUN_ENV=(env UNO_SWEBENCH_BACKEND=1)
  fi

  CMD=(
    python -m eval_pipeline
    --router planner
    --bench "${BENCH}"
    --api_key "${API_KEY}"
    --api_base "${API_BASE}"
    --local_base "${LOCAL_BASE}"
    --local_model "${MODEL_ID}"
    --router_model "${MODEL_ID}"
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
    printf '%q ' "${RUN_ENV[@]}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
  else
    echo "[RUN] ${MODEL_NAME} ${BENCH}"
    if [[ "${BENCH}" == "swebench" ]]; then
      echo "[MODE] swebench planning route (UNO_SWEBENCH_BACKEND=1)"
    fi
    "${RUN_ENV[@]}" "${CMD[@]}" 2>&1 | tee "${OUT_DIR}.log"
  fi
done

python scripts/collect_results.py --root "${EVAL_OUT}" --format md
