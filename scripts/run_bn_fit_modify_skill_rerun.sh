#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

TRAJECTORY_A="${TRAJECTORY_A:-data/eval/deepseek_v4_pro_planner_terminalbench_20260625_1524/logs/attempt_0/bn-fit-modify/trajectory.json}"
TRAJECTORY_B="${TRAJECTORY_B:-data/eval/deepseek_v4_pro_planner_terminalbench_20260625_1524/logs/attempt_0/bn-fit-modify/trajectory.json}"

SKILLS_PATH="${SKILLS_PATH:-${PROJECT_DIR}/terminal_bench_skills_init.json}"
TOP_K="${TOP_K:-2}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_STEPS="${MAX_STEPS:-30}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-300}"

export SUBAGENT_ENABLE_SKILLS="${SUBAGENT_ENABLE_SKILLS:-0}"
export SUBAGENT_SKILLS_TOP_K="${SUBAGENT_SKILLS_TOP_K:-${TOP_K}}"
export SUBAGENT_SKILLS_MODE="${SUBAGENT_SKILLS_MODE:-task_only}"
export SUBAGENT_SKILLS_MODELS="${SUBAGENT_SKILLS_MODELS:-${MODEL}}"
export TERMINAL_BENCH_SKILLS_PATH="${TERMINAL_BENCH_SKILLS_PATH:-${SKILLS_PATH}}"
export UNO_POOLS_PATH="${UNO_POOLS_PATH:-${PROJECT_DIR}/configs/pools.deepseek_v32.yaml}"

BASE_OUT="${BASE_OUT:-data/eval/deepseek_v4_pro_planner_terminalbench_20260625_1524/skill_reruns}"
mkdir -p "${BASE_OUT}"
LOG_FILE="${LOG_FILE:-${BASE_OUT}/run.log}"

{
  echo "========================================"
  echo "Terminal-Bench skill rerun"
  echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Project: ${PROJECT_DIR}"
  echo "Model: ${MODEL}"
  echo "Skills: ${TERMINAL_BENCH_SKILLS_PATH}"
  echo "Top-k skills: ${SUBAGENT_SKILLS_TOP_K}"
  echo "Resume from commands checkpoint: enabled"
  echo "Output root: ${BASE_OUT}"
  echo "Log file: ${LOG_FILE}"
  echo "========================================"
} > "${LOG_FILE}"

{
  echo
  echo "---- Rerun call_00_ldkrVwsMtC0cCmC8yUkv0010 ----"
  python scripts/rerun_terminalbench_from_commands_checkpoint.py \
    --trajectory "${TRAJECTORY_A}" \
    --tool-call-id "call_00_ldkrVwsMtC0cCmC8yUkv0010" \
    --model "${MODEL}" \
    --skills "${SKILLS_PATH}" \
    --top-k "${TOP_K}" \
    --max-steps "${MAX_STEPS}" \
    --command-timeout "${COMMAND_TIMEOUT}" \
    --run-tests \
    --output-dir "${BASE_OUT}/bn-fit-modify_call_00_ldkrVwsMtC0cCmC8yUkv0010"

  echo
  echo "---- Rerun call_00_CPGx5QlV8heWg10ziod14855 ----"
  python scripts/rerun_terminalbench_from_commands_checkpoint.py \
    --trajectory "${TRAJECTORY_B}" \
    --tool-call-id "call_00_CPGx5QlV8heWg10ziod14855" \
    --model "${MODEL}" \
    --skills "${SKILLS_PATH}" \
    --top-k "${TOP_K}" \
    --max-steps "${MAX_STEPS}" \
    --command-timeout "${COMMAND_TIMEOUT}" \
    --run-tests \
    --output-dir "${BASE_OUT}/bn-fit-modify_call_00_CPGx5QlV8heWg10ziod14855"

  echo
  echo "Done. Results under: ${BASE_OUT}"
  echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
} >> "${LOG_FILE}" 2>&1

echo "Done. Log: ${LOG_FILE}"
