#!/usr/bin/env bash
set -euo pipefail

IMAGE_PREFIX="${TERMINALBENCH_DOCKER_IMAGE_PREFIX:-alexgshaw}"
IMAGE_TAG="${TERMINALBENCH_DOCKER_IMAGE_TAG:-20251031}"

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

if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found. Install Docker Desktop and ensure docker is on PATH." >&2
    exit 2
fi

if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not reachable. Start Docker Desktop and retry." >&2
    exit 2
fi

echo "Installing ${#TASK_IDS[@]} Terminal-Bench Docker images: ${IMAGE_PREFIX}/<task>:${IMAGE_TAG}"
for task_id in "${TASK_IDS[@]}"; do
    image="${IMAGE_PREFIX}/${task_id}:${IMAGE_TAG}"
    if docker image inspect "${image}" >/dev/null 2>&1; then
        echo "Already installed: ${image}"
        continue
    fi
    echo "Pulling: ${image}"
    docker pull "${image}"
done

echo "Done."
