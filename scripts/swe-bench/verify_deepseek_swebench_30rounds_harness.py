#!/usr/bin/env python
"""Run SWE-Bench harness on DeepSeek 30-round router-planning predictions.

This is the 30-round counterpart of verify_deepseek_swebench_8rounds_harness.py.
It reads eval_pipeline predictions.jsonl, extracts clean unified diffs, writes a
conversion report, and invokes the official SWE-Bench harness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from verify_deepseek_swebench_8rounds_harness import convert_predictions


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "eval"
    / "deepseek_planner"
    / "swebench_router_planning_30rounds"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "eval"
    / "deepseek_planner"
    / "swebench_router_planning_30rounds_harness"
)
DEFAULT_HF_HOME = REPO_ROOT / "data" / "huggingface"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", default="deepseek_router_planning_30rounds_harness")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "2")))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--cache-level", default="instance")
    parser.add_argument("--conda-env", default=os.environ.get("SWEBENCH_CONDA_ENV", "swebench"))
    parser.add_argument("--online", action="store_true", help="Allow HF Hub network access.")
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Submit invalid/non-diff text as-is. Default writes an empty patch for invalid predictions.",
    )
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    work_dir = args.work_dir.resolve()
    harness_predictions = work_dir / "predictions.swebench.jsonl"
    conversion_report = work_dir / "conversion_report.json"
    report_dir = work_dir / "reports"

    rows, report = convert_predictions(predictions_path, keep_invalid=args.keep_invalid)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    with harness_predictions.open("w", encoding="utf-8") as f:
        for row in rows:
            row["model_name_or_path"] = "deepseek_router_planning_30rounds"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    conversion_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    env.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    if not args.online:
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")

    python_cmd = [sys.executable]
    if args.conda_env:
        python_cmd = ["conda", "run", "-n", args.conda_env, "python"]

    cmd = [
        *python_cmd,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--predictions_path",
        str(harness_predictions),
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
    print(f"Harness predictions: {harness_predictions}")
    print(f"Conversion report: {conversion_report}")
    print(f"Report dir: {report_dir}")
    print(f"Instances: {len(rows)}")
    print(f"Clean diffs: {report['summary']['clean_diff']}")
    print(f"Invalid/empty submitted as empty patch: {report['summary']['invalid_or_empty']}")
    print("Command:")
    print(" ".join(cmd))

    stdout_path = work_dir / "swebench_harness_stdout.log"
    stderr_path = work_dir / "swebench_harness_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=report_dir, env=env, stdout=out, stderr=err, text=True)
    print(f"Harness stdout: {stdout_path}")
    print(f"Harness stderr: {stderr_path}")
    print(f"Harness report dir: {report_dir}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
