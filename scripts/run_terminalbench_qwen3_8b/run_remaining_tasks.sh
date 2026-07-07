#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASK_IDS=(
    hf-model-inference
    install-windows-3.11
    large-scale-text-editing
    mteb-leaderboard
    mteb-retrieve
    polyglot-c-py
    polyglot-rust-c
    protein-assembly
    prove-plus-comm
    pypi-server
    pytorch-model-cli
    pytorch-model-recovery
    qemu-alpine-ssh
    qemu-startup
    query-optimize
    raman-fitting
    regex-chess
    regex-log
    reshard-c4-data
    rstan-to-pystan
    sam-cell-seg
    sanitize-git-repo
    schemelike-metacircular-eval
    sparql-university
    sqlite-db-truncate
    sqlite-with-gcov
    torch-pipeline-parallelism
    torch-tensor-parallelism
    train-fasttext
    tune-mjcf
    video-processing
    vulnerable-secret
    winning-avg-corewars
    write-compressor
)

export TERMINALBENCH_TASK_IDS="${TASK_IDS[*]}"
export TERMINALBENCH_PREPULL_IMAGES="${TERMINALBENCH_PREPULL_IMAGES:-0}"
export MAX_TASKS=""
export RUN_NAME="${RUN_NAME:-deepseek_v4_pro_qwen3_8b_router_terminalbench_remaining_tasks}"

exec "${SCRIPT_DIR}/../run_terminalbench_qwen3_8b_fixed_routing.sh"
