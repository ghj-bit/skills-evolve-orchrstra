#!/usr/bin/env python
"""Run SWE-Bench harness on DeepSeek 8-round router-planning predictions.

This script is intentionally separate from generation.  It converts
eval_pipeline predictions.jsonl into the official SWE-Bench predictions schema,
tries to extract a clean unified diff from each prediction, writes a conversion
report, and then invokes ``swebench.harness.run_evaluation``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data"
    / "eval"
    / "deepseek_planner"
    / "swebench_router_planning_8rounds"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "eval"
    / "deepseek_planner"
    / "swebench_router_planning_8rounds_harness"
)
DEFAULT_HF_HOME = REPO_ROOT / "data" / "huggingface"


FINAL_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
FENCED_DIFF_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-id", default="deepseek_router_planning_8rounds_harness")
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


def convert_predictions(path: Path, keep_invalid: bool) -> tuple[list[dict], dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    instances: list[dict] = []
    clean_count = 0
    invalid_count = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            instance_id = raw.get("instance_id") or raw.get("task_id")
            if not instance_id:
                raise ValueError(f"Missing task_id/instance_id at {path}:{line_no}")
            source_text = str(raw.get("answer") or raw.get("model_patch") or "")
            full_trace = str(raw.get("full_trace") or "")
            patch, reason = extract_clean_patch(source_text, full_trace)
            clean = reason == "clean_diff"
            if clean:
                clean_count += 1
            else:
                invalid_count += 1
                if not keep_invalid:
                    patch = ""
            rows.append(
                {
                    "instance_id": str(instance_id),
                    "model_name_or_path": str(raw.get("model_name_or_path") or "deepseek_router_planning_8rounds"),
                    "model_patch": patch,
                }
            )
            instances.append(
                {
                    "instance_id": str(instance_id),
                    "source_answer_chars": len(source_text),
                    "submitted_patch_chars": len(patch),
                    "status": reason,
                    "route_count": raw.get("route_count", 0),
                    "routed_backends": raw.get("routed_backends", []),
                }
            )
    return rows, {
        "source": str(path),
        "summary": {
            "total": len(rows),
            "clean_diff": clean_count,
            "invalid_or_empty": invalid_count,
            "keep_invalid": keep_invalid,
        },
        "instances": instances,
    }


def extract_clean_patch(answer: str, full_trace: str) -> tuple[str, str]:
    candidates: list[str] = []
    candidates.extend(_final_answers(answer))
    candidates.extend(_final_answers(full_trace))
    candidates.append(answer)
    candidates.extend(_fenced_blocks(answer))
    candidates.extend(_fenced_blocks(full_trace))
    candidates.extend(_docker_diffs(full_trace))

    for candidate in candidates:
        patch = _strip_to_diff(candidate)
        if not patch:
            continue
        if _is_clean_unified_diff(patch):
            return patch, "clean_diff"
    if answer.strip():
        return answer, "not_a_clean_unified_diff"
    return "", "empty"


def _final_answers(text: str) -> list[str]:
    return [m.group(1).strip() for m in FINAL_RE.finditer(text or "")]


def _fenced_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in FENCED_DIFF_RE.finditer(text or "")]


def _docker_diffs(text: str) -> list[str]:
    marker = "[SWE-BENCH DOCKER DIFF]"
    end = "[/SWE-BENCH DOCKER DIFF]"
    out: list[str] = []
    start = 0
    while True:
        i = text.find(marker, start)
        if i < 0:
            return out
        j = text.find(end, i + len(marker))
        if j < 0:
            return out
        out.append(text[i + len(marker):j].strip())
        start = j + len(end)


def _strip_to_diff(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    starts = [idx for idx in (value.find("diff --git "), value.find("--- a/")) if idx >= 0]
    if not starts:
        return ""
    value = value[min(starts):].strip()
    stop_markers = ["\n```", "\n</obs>", "\n[/TOOL]", "\n[ASSISTANT]", "\n<plan ", "\n<route "]
    stops = [value.find(marker) for marker in stop_markers if value.find(marker) >= 0]
    if stops:
        value = value[: min(stops)].rstrip()
    return value


def _is_clean_unified_diff(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(marker in stripped for marker in ["```", "<obs", "</obs>", "[ASSISTANT]", "[TOOL]", "<plan", "<route"]):
        return False
    if "--- " not in stripped or "+++ " not in stripped or "@@ " not in stripped:
        return False
    return stripped.startswith("diff --git ") or stripped.startswith("--- a/")


if __name__ == "__main__":
    raise SystemExit(main())
