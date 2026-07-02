#!/usr/bin/env python
"""Run deepseek-v4-flash as a direct flat Terminal-Bench baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("UNO_POOLS_PATH", str(ROOT / "configs" / "pools.deepseek_v32.yaml"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_pools  # noqa: E402
from eval_pipeline.benchmarks.terminalbench import TerminalBench  # noqa: E402
from eval_pipeline.routers.direct import DirectRouter  # noqa: E402


MODEL_ID = "deepseek-v4-flash"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _model_cfg(model_id: str) -> dict:
    pools = load_pools()
    for model in pools["raw"].get("models", []):
        if model.get("id") == model_id:
            return model
    raise SystemExit(f"Model not found in pool: {model_id}")


def _read_trajectory_usage(path: Path) -> dict:
    usage = {
        "route_count": 0,
        "routed_models": [MODEL_ID],
        "routed_skills": ["execute_command"],
        "routed_backends": ["direct_flat"],
        "cost": 0.0,
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    if not path.exists():
        usage["stats_error"] = f"trajectory not found: {path}"
        return usage
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        usage["stats_error"] = f"failed to read trajectory: {exc}"
        return usage
    for key in usage:
        if key in data:
            usage[key] = data[key]
    return usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--cmd-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    cfg = _model_cfg(MODEL_ID)
    api_base = cfg.get("api_base") or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
    api_key = cfg.get("api_key") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY") or "EMPTY"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.output_dir or ROOT / "data" / "eval" / f"direct_deepseek_v4_flash_terminalbench_{timestamp}")
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "predictions.jsonl"
    verify_file = out / "verification.jsonl"
    logs_dir = out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and verify_file.exists():
        for line in verify_file.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["task_id"])
            except Exception:
                pass

    tb = TerminalBench(subagent_max_steps=args.max_steps, subagent_cmd_timeout=args.cmd_timeout)
    tasks = tb.load(args.max_tasks)
    if args.task_ids:
        wanted = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [task for task in tasks if task.task_id in wanted]
    tasks = [task for task in tasks if task.task_id not in done]

    router = DirectRouter(model_id=MODEL_ID, api_base=api_base, api_key=api_key)

    print("=" * 60, flush=True)
    print("Terminal-Bench direct flat baseline", flush=True)
    print(f"Model:      {MODEL_ID}", flush=True)
    print(f"API base:   {api_base}", flush=True)
    print(f"Output:     {out}", flush=True)
    print(f"Tasks:      {len(tasks)}", flush=True)
    print(f"Max steps:  {args.max_steps}", flush=True)
    print("=" * 60, flush=True)

    started = time.time()
    records = []
    with pred_file.open("a", encoding="utf-8") as pred_out, verify_file.open("a", encoding="utf-8") as ver_out:
        for idx, task in enumerate(tasks, start=1):
            t0 = time.time()
            print(f"[{_now_iso()}] [{idx}/{len(tasks)}] {task.task_id}", flush=True)
            res = tb.run_interactive(task, router, logs_dir=str(logs_dir), flat_mode=True)
            trajectory_path = logs_dir / task.task_id / "trajectory.json"
            usage = _read_trajectory_usage(trajectory_path)
            reward = float(res.reward or 0.0)
            pred = {
                "task_id": task.task_id,
                "answer": "(interactive)",
                "trajectory_path": str(trajectory_path),
                "reward": reward,
                "error": res.error,
                **usage,
            }
            ver = {
                "task_id": task.task_id,
                "reward": reward,
                "error": res.error,
                "log": (res.log or "")[:500],
                "time_seconds": round(time.time() - t0, 3),
            }
            pred_out.write(json.dumps(pred, ensure_ascii=False) + "\n")
            pred_out.flush()
            ver_out.write(json.dumps(ver, ensure_ascii=False) + "\n")
            ver_out.flush()
            records.append(pred)
            print(
                f"  reward={reward:.1f} cost=${pred.get('cost', 0):.6f} "
                f"time={ver['time_seconds']:.1f}s error={res.error or ''}",
                flush=True,
            )

    total = len(records)
    passed = sum(1 for r in records if float(r.get("reward", 0) or 0) > 0)
    total_cost = sum(float(r.get("cost", 0) or 0) for r in records)
    total_tokens = sum(int(r.get("tokens", 0) or 0) for r in records)
    prompt_tokens = sum(int(r.get("prompt_tokens", 0) or 0) for r in records)
    completion_tokens = sum(int(r.get("completion_tokens", 0) or 0) for r in records)
    summary = {
        "router": f"Direct({MODEL_ID})",
        "benchmark": tb.name,
        "mode": "direct_flat",
        "total": total,
        "passed": passed,
        "success_rate": round(passed / max(total, 1), 6),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / max(total, 1), 6),
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": round(time.time() - started, 3),
        "predictions": str(pred_file),
        "verification": str(verify_file),
        "logs": str(logs_dir),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
