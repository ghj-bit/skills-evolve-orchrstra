#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi

EVAL_DIR="${EVAL_DIR:-${PROJECT_DIR}/data/eval/deepseek_v4_pro_qwen3_8b_router_terminalbench}"
OUTPUT="${OUTPUT:-${PROJECT_DIR}/terminal_bench_skills_gen.json}"
EXISTING_SKILLS="${EXISTING_SKILLS:-${PROJECT_DIR}/terminal_bench_skills_init.json}"
PROMPT_DIR="${PROMPT_DIR:-${EVAL_DIR}/skill_distill_prompts}"
SAVE_PROMPTS="${SAVE_PROMPTS:-0}"
ATTEMPTS="${ATTEMPTS:-attempt_0,attempt_1}"
TASKS="${TASKS:-}"
MAX_TRAJECTORIES="${MAX_TRAJECTORIES:-0}"
MAX_DELEGATES="${MAX_DELEGATES:-8}"
COMMANDS_CHARS="${COMMANDS_CHARS:-16000}"
TEST_OUTPUT_CHARS="${TEST_OUTPUT_CHARS:-12000}"
CTRF_CHARS="${CTRF_CHARS:-12000}"

MODEL="${MODEL:-${DEEPSEEK_MODEL_ID:-deepseek-v4-flash}}"
API_BASE="${API_BASE:-${DEEPSEEK_API_BASE:-https://api.deepseek.com}}"
API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-}}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TIMEOUT="${TIMEOUT:-120}"
RETRIES="${RETRIES:-3}"

MERGE_EXISTING="${MERGE_EXISTING:-0}"
NORMAL_ONLY="${NORMAL_ONLY:-1}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
    python "${PROJECT_DIR}/scripts/distill_terminalbench_skills.py"
    --eval-dir "${EVAL_DIR}"
    --output "${OUTPUT}"
    --existing-skills "${EXISTING_SKILLS}"
    --attempts "${ATTEMPTS}"
    --max-trajectories "${MAX_TRAJECTORIES}"
    --max-delegates "${MAX_DELEGATES}"
    --commands-chars "${COMMANDS_CHARS}"
    --test-output-chars "${TEST_OUTPUT_CHARS}"
    --ctrf-chars "${CTRF_CHARS}"
    --model "${MODEL}"
    --api-base "${API_BASE}"
    --temperature "${TEMPERATURE}"
    --max-tokens "${MAX_TOKENS}"
    --timeout "${TIMEOUT}"
    --retries "${RETRIES}"
)

if [[ "${SAVE_PROMPTS}" != "0" ]]; then
    CMD+=(--prompt-dir "${PROMPT_DIR}")
fi

if [[ -n "${TASKS}" ]]; then
    CMD+=(--tasks "${TASKS}")
fi

if [[ "${MERGE_EXISTING}" != "0" ]]; then
    CMD+=(--merge-existing)
fi

if [[ "${NORMAL_ONLY}" == "0" ]]; then
    CMD+=(--no-normal-only)
else
    CMD+=(--normal-only)
fi

if [[ "${DRY_RUN}" != "0" ]]; then
    CMD+=(--dry-run)
else
    CMD+=(--api-key "${API_KEY}")
fi

echo "========================================"
echo "Terminal-Bench Skill Distillation"
echo "Project:          ${PROJECT_DIR}"
echo "Eval dir:         ${EVAL_DIR}"
echo "Output:           ${OUTPUT}"
echo "Existing skills:  ${EXISTING_SKILLS}"
echo "Save prompts:     ${SAVE_PROMPTS}"
echo "Prompt dir:       ${PROMPT_DIR}"
echo "Attempts:         ${ATTEMPTS}"
echo "Tasks:            ${TASKS}"
echo "Normal only:      ${NORMAL_ONLY}"
echo "Merge existing:   ${MERGE_EXISTING}"
echo "Dry run:          ${DRY_RUN}"
echo "Model:            ${MODEL}"
echo "API base:         ${API_BASE}"
echo "Max trajectories: ${MAX_TRAJECTORIES}"
echo "Commands chars:   ${COMMANDS_CHARS}"
echo "Test output chars:${TEST_OUTPUT_CHARS}"
echo "CTRF chars:       ${CTRF_CHARS}"
echo "========================================"

"${CMD[@]}"
