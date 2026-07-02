#!/usr/bin/env bash
# Planner SWE-Bench runner for deepseek-v3.2.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

MODEL_NAME="${MODEL_NAME:-deepseek}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash}"
DEEPSEEK_BASE="${DEEPSEEK_BASE:-https://api.deepseek.com}"
LOCAL_BASE="${LOCAL_BASE:-${DEEPSEEK_BASE}}"
API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-}}"
API_BASE="${API_BASE:-${DEEPSEEK_BASE}}"

if [[ -z "${API_KEY}" ]]; then
  echo "API_KEY or DEEPSEEK_API_KEY must be set before running this script." >&2
  exit 2
fi

UNO_POOLS_PATH="${UNO_POOLS_PATH:-${PROJECT_DIR}/configs/pools.deepseek_v32.yaml}"
UNO_SYSTEM_PROMPT="${UNO_SYSTEM_PROMPT:-${PROJECT_DIR}/configs/uno/system_prompt_deepseek_v32.txt}"
export UNO_POOLS_PATH UNO_SYSTEM_PROMPT

EVAL_OUT="${EVAL_OUT:-${PROJECT_DIR}/data/eval}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/data/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_DIR}/data/huggingface/datasets}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_DIR}/data/huggingface/hub}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE HF_DATASETS_OFFLINE HF_HUB_OFFLINE

GEN_WORKERS="${GEN_WORKERS:-2}"
VERIFY_WORKERS="${VERIFY_WORKERS:-2}"
PASS_K="${PASS_K:-1}"
UNO_VERBOSE_RESPONSES="${UNO_VERBOSE_RESPONSES:-1}"
UNO_MAX_ROUNDS="${UNO_MAX_ROUNDS:-8}"
export UNO_VERBOSE_RESPONSES UNO_MAX_ROUNDS

MAX_TASKS_ARG=()
SWE_MODE="${SWE_MODE:-interactive}"
OUT_SUFFIX="${OUT_SUFFIX:-}"
DRY_RUN=false

for cert_var in SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE; do
  cert_path="${!cert_var:-}"
  if [[ -n "${cert_path}" && ! -f "${cert_path}" ]]; then
    echo "Unset ${cert_var}: file not found (${cert_path})"
    unset "${cert_var}"
  fi
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-tasks)
      MAX_TASKS_ARG=(--max_tasks "$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --router-planning)
      SWE_MODE="planning"
      shift
      ;;
    --interactive-mode)
      SWE_MODE="interactive"
      shift
      ;;
    --out-suffix)
      OUT_SUFFIX="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

BENCH="swebench"
BENCH_EXTRA=()
OUT_BENCH="${BENCH}"
if [[ "${SWE_MODE}" == "planning" ]]; then
  OUT_BENCH="${BENCH}_router_planning"
  UNO_SWEBENCH_BACKEND="${UNO_SWEBENCH_BACKEND:-1}"
elif [[ "${SWE_MODE}" == "interactive" ]]; then
  BENCH_EXTRA=(--interactive)
  UNO_SWEBENCH_BACKEND="${UNO_SWEBENCH_BACKEND:-0}"
else
  echo "Invalid SWE_MODE: ${SWE_MODE}. Use 'interactive' or 'planning'."
  exit 1
fi
export UNO_SWEBENCH_BACKEND
if [[ -n "${OUT_SUFFIX}" ]]; then
  OUT_BENCH="${OUT_BENCH}_${OUT_SUFFIX}"
fi
OUT_DIR="${EVAL_OUT}/${MODEL_NAME}_planner/${OUT_BENCH}"

echo "Model: ${MODEL_NAME}"
echo "Router endpoint: ${LOCAL_BASE}"
echo "Worker endpoint: ${API_BASE}"
echo "Benchmark: ${BENCH}"
echo "SWE mode: ${SWE_MODE}"
echo "Output dir: ${OUT_DIR}"
echo "Uno pool: ${UNO_POOLS_PATH}"
echo "Uno prompt: ${UNO_SYSTEM_PROMPT}"
echo "HF cache: ${HF_HOME}"
echo "HF offline: datasets=${HF_DATASETS_OFFLINE}, hub=${HF_HUB_OFFLINE}"
echo "Workers: gen=${GEN_WORKERS}, verify=${VERIFY_WORKERS}"
echo "Uno max rounds: ${UNO_MAX_ROUNDS}"
echo "SWE route backend: ${UNO_SWEBENCH_BACKEND}"
if [[ ${#MAX_TASKS_ARG[@]} -eq 0 ]]; then
  echo "Max tasks: all"
else
  echo "Max tasks: ${MAX_TASKS_ARG[*]}"
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
  echo "[SKIP] ${MODEL_NAME} ${BENCH}: ${OUT_DIR}/summary.json already exists"
  exit 0
fi

mkdir -p "$(dirname "${OUT_DIR}")"

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[DRY] '
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  echo "[RUN] ${MODEL_NAME} ${BENCH}"
  START_TS="$(date +%s)"
  START_ISO="$(date -Iseconds)"
  echo "Start time: ${START_ISO}"
  mkdir -p "${OUT_DIR}"
  "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/run.log"
  END_TS="$(date +%s)"
  END_ISO="$(date -Iseconds)"
  ELAPSED_SEC=$((END_TS - START_TS))
  {
    echo "start_time=${START_ISO}"
    echo "end_time=${END_ISO}"
    echo "elapsed_seconds=${ELAPSED_SEC}"
    printf 'elapsed_hms=%02d:%02d:%02d\n' "$((ELAPSED_SEC / 3600))" "$(((ELAPSED_SEC % 3600) / 60))" "$((ELAPSED_SEC % 60))"
    echo "uno_max_rounds=${UNO_MAX_ROUNDS}"
    echo "swe_mode=${SWE_MODE}"
    echo "output_dir=${OUT_DIR}"
  } | tee "${OUT_DIR}/timing.txt"
fi

python scripts/collect_results.py --root "${EVAL_OUT}" --format md
