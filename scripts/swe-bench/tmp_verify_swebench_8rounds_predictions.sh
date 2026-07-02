#!/usr/bin/env bash
# Verify 8-round DeepSeek router-planning SWE-Bench predictions with official harness.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

MAX_WORKERS="${MAX_WORKERS:-2}"

python scripts/tmp_verify_swebench_8rounds_predictions.py \
  --max-workers "${MAX_WORKERS}" \
  "$@"
