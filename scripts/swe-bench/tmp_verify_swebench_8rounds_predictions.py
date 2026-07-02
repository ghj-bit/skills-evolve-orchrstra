#!/usr/bin/env python
"""Verify 8-round DeepSeek router-planning SWE-Bench predictions.

Input:
  data/eval/deepseek_planner/swebench_router_planning_8rounds/predictions.jsonl

Output:
  data/eval/deepseek_planner/swebench_router_planning_8rounds_official/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    predictions = (
        REPO_ROOT
        / "data"
        / "eval"
        / "deepseek_planner"
        / "swebench_router_planning_8rounds"
        / "predictions.jsonl"
    )
    work_dir = (
        REPO_ROOT
        / "data"
        / "eval"
        / "deepseek_planner"
        / "swebench_router_planning_8rounds_official"
    )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tmp_verify_swebench_predictions.py"),
        "--predictions",
        str(predictions),
        "--work-dir",
        str(work_dir),
        "--run-id",
        "deepseek_router_planning_8rounds_official",
    ]
    cmd.extend(sys.argv[1:])

    print("Verifying 8-round SWE-Bench predictions")
    print(f"Input predictions: {predictions}")
    print(f"Output work dir: {work_dir}")
    print("Command:")
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
