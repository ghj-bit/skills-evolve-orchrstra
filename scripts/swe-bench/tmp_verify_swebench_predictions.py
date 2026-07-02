#!/usr/bin/env python
"""Verify existing eval_pipeline SWE-Bench predictions with SWE-Bench harness.

This converts eval_pipeline predictions:
  {"task_id": "...", "answer": "..."}

into SWE-Bench harness predictions:
  {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}

Then it calls:
  python -m swebench.harness.run_evaluation ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "eval" / "deepseek_planner" / "swebench_router_planning" / "predictions.jsonl"
DEFAULT_WORK_DIR = REPO_ROOT / "data" / "eval" / "deepseek_planner" / "swebench_router_planning_official"
DEFAULT_HF_HOME = REPO_ROOT / "data" / "huggingface"


def _read_eval_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id") or row.get("task_id")
            model_patch = row.get("model_patch") or row.get("answer")
            if not instance_id:
                raise ValueError(f"Missing task_id/instance_id at {path}:{line_no}")
            if model_patch is None:
                raise ValueError(f"Missing answer/model_patch at {path}:{line_no}")
            rows.append(
                {
                    "instance_id": str(instance_id),
                    "model_name_or_path": str(row.get("model_name_or_path") or "deepseek_router_planning"),
                    "model_patch": str(model_patch),
                }
            )
    return rows


def _write_harness_predictions(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SWE-Bench official harness on an existing predictions.jsonl")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", default="deepseek_router_planning_official")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("VERIFY_WORKERS", "2")))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--cache-level", default="instance")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_false", dest="offline")
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    work_dir = args.work_dir.resolve()
    harness_pred_path = work_dir / "predictions.swebench.jsonl"
    report_dir = work_dir / "reports"

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    rows = _read_eval_predictions(predictions_path)
    if not rows:
        raise ValueError(f"No predictions found in {predictions_path}")
    _write_harness_predictions(rows, harness_pred_path)

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    env.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    if args.offline:
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")

    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        str(harness_pred_path),
        "--max_workers",
        str(args.max_workers),
        "--run_id",
        args.run_id,
        "--timeout",
        str(args.timeout),
        "--cache_level",
        args.cache_level,
        "--report_dir",
        str(report_dir),
    ]

    print(f"Input predictions: {predictions_path}")
    print(f"Harness predictions: {harness_pred_path}")
    print(f"Report dir: {report_dir}")
    print(f"Instances: {len(rows)}")
    print("Command:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd, env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
