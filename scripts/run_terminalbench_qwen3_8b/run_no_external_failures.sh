#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TASK_IDS=(
    adaptive-rejection-sampler
    bn-fit-modify
    break-filter-js-from-html
    build-cython-ext
    build-pmars
    build-pov-ray
    caffe-cifar-10
    cancel-async-tasks
    chess-best-move
    circuit-fibsqrt
    cobol-modernization
    code-from-image
    compile-compcert
    configure-git-webserver
    constraints-scheduling
    count-dataset-tokens
    crack-7z-hash
    custom-memory-heap-crash
    db-wal-recovery
    distribution-search
    dna-assembly
    dna-insert
    extract-elf
    extract-moves-from-video
    feal-differential-cryptanalysis
    feal-linear-cryptanalysis
    filter-js-from-html
    financial-document-processor
    fix-code-vulnerability
    fix-git
    fix-ocaml-gc
    gcode-to-text
    git-leak-recovery
    git-multibranch
    gpt2-codegolf
    headless-terminal
    kv-store-grpc
    largest-eigenval
    llm-inference-batching-scheduler
    log-summary-date-ranges
    mailman
    make-doom-for-mips
    make-mips-interpreter
    mcmc-sampling-stan
    merge-diff-arc-agi-task
    model-extraction-relu-logits
    modernize-scientific-stack
    multi-source-data-merger
    nginx-request-logging
    openssl-selfsigned-cert
    overfull-hbox
    password-recovery
    path-tracing
    path-tracing-reverse
    portfolio-optimization
)

export TERMINALBENCH_TASK_IDS="${TASK_IDS[*]}"
export MAX_TASKS=""
export RUN_NAME="${RUN_NAME:-deepseek_v4_pro_qwen3_8b_router_terminalbench_no_external_failures}"

exec bash "${SCRIPT_DIR}/../run_terminalbench_qwen3_8b_fixed_routing.sh"
