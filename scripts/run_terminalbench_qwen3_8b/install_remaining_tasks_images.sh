#!/usr/bin/env bash
set -euo pipefail

IMAGE_PREFIX="${TERMINALBENCH_DOCKER_IMAGE_PREFIX:-alexgshaw}"
IMAGE_TAG="${TERMINALBENCH_DOCKER_IMAGE_TAG:-20251031}"

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
