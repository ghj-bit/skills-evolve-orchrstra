#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.env"
    set +a
fi

PYTHON="${PYTHON:-python}"
TOOLBENCH_DATASET="${TOOLBENCH_DATASET:-Team-ACE/ToolACE}"
TOOLBENCH_SPLIT="${TOOLBENCH_SPLIT:-train}"
TOOLBENCH_DATA_DIR="${TOOLBENCH_DATA_DIR:-${PROJECT_DIR}/data/toolbench/ToolACE}"

mkdir -p "$(dirname "${TOOLBENCH_DATA_DIR}")"

echo "Downloading ToolBench dataset"
echo "Dataset: ${TOOLBENCH_DATASET}"
echo "Split:   ${TOOLBENCH_SPLIT}"
echo "Output:  ${TOOLBENCH_DATA_DIR}"

"${PYTHON}" - "${TOOLBENCH_DATASET}" "${TOOLBENCH_SPLIT}" "${TOOLBENCH_DATA_DIR}" <<'PY'
import sys
from pathlib import Path

from datasets import load_dataset

dataset_name, split, output_dir = sys.argv[1:4]
output_path = Path(output_dir)

ds = load_dataset(dataset_name, split=split)
output_path.parent.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(output_path))

print(f"Saved {len(ds)} rows to {output_path}")
PY

echo "Done. Use:"
echo "  TOOLBENCH_DATA_DIR=${TOOLBENCH_DATA_DIR} bash scripts/run_toolbench_qwen3_8b_fixed_routing.sh"
