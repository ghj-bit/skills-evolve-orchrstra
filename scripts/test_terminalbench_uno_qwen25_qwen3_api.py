#!/usr/bin/env python
"""Smoke test the router and worker APIs used by Terminal-Bench fixed routing.

This mirrors the defaults in:
    scripts/run_terminalbench_uno_qwen25_7b_qwen3_8b_fixed_routing.sh

It checks:
  1. router/planner endpoint with a normal chat completion
  2. router/planner endpoint with OpenAI tools, matching Terminal-Bench planner calls
  3. Qwen3-8B worker endpoint with a normal chat completion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTER_API_BASE = (
    "https://notebook-inspire.sii.edu.cn/"
    "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/"
    "project-b795c114-135a-40db-b3d0-19b60f25237b/"
    "user-543feed4-0be2-4972-8987-a324af06c93f/"
    "vscode/3a8e9a70-c91e-459d-ad61-e9b54493df6c/"
    "e5808f80-5446-4406-b9f2-18f284c91563/"
    "proxy/8000/v1"
)
DEFAULT_ROUTER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_WORKER_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_WORKER_MODEL = "Qwen/Qwen3-8B"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-api-base", default=os.environ.get("UNO_ROUTER_API_BASE", DEFAULT_ROUTER_API_BASE))
    parser.add_argument("--router-api-key", default=os.environ.get("UNO_ROUTER_API_KEY", "empty"))
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PLANNER_MODEL_ID", DEFAULT_ROUTER_MODEL),
        help="Model used for Terminal-Bench planner chat calls.",
    )
    parser.add_argument(
        "--router-model",
        default=os.environ.get("ROUTER_MODEL_ID", DEFAULT_ROUTER_MODEL),
        help="Model used for router calls. Defaults to the same served model.",
    )
    parser.add_argument(
        "--worker-api-base",
        default=os.environ.get("QWEN_API_BASE") or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_WORKER_API_BASE),
    )
    parser.add_argument(
        "--worker-api-key",
        default=os.environ.get("QWEN_API_KEY") or os.environ.get("SILICONFLOW_API_KEY", ""),
    )
    parser.add_argument(
        "--worker-model",
        default=os.environ.get("WORKER_MODEL_ID") or os.environ.get("OPENROUTER_MODEL", DEFAULT_WORKER_MODEL),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--skip-worker", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    return parser.parse_args()


def print_config(args: argparse.Namespace) -> None:
    print("Router/planner endpoint:")
    print(f"  base_url: {args.router_api_base}")
    print(f"  planner_model: {args.planner_model}")
    print(f"  router_model:  {args.router_model}")
    print(f"  api_key: {'<empty>' if not args.router_api_key else '<set>'}")
    print("Worker endpoint:")
    print(f"  base_url: {args.worker_api_base}")
    print(f"  model:    {args.worker_model}")
    print(f"  api_key:  {'<empty>' if not args.worker_api_key else '<set>'}")


def post_chat_completion(api_base: str, api_key: str, payload: dict) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def chat_test(name: str, api_base: str, api_key: str, model: str, max_tokens: int) -> bool:
    print(f"\n[{name}] normal chat")
    try:
        resp = post_chat_completion(
            api_base,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Reply with exactly: pong"},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    msg = resp.get("choices", [{}])[0].get("message", {})
    print(f"OK: {msg.get('content')!r}")
    print(f"usage: {resp.get('usage')}")
    return True


def tools_test(api_base: str, api_key: str, model: str, max_tokens: int) -> bool:
    print("\n[router/planner] OpenAI tools chat")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate a concrete subtask to a worker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_model": {"type": "string", "enum": [DEFAULT_WORKER_MODEL]},
                        "instruction": {"type": "string"},
                    },
                    "required": ["worker_model", "instruction"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Declare the task complete.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        },
    ]
    try:
        resp = post_chat_completion(
            api_base,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a Terminal-Bench planner."},
                    {"role": "user", "content": "Plan one simple shell check, or submit if done."},
                ],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False

    msg = resp.get("choices", [{}])[0].get("message", {})
    print(f"OK content: {msg.get('content')!r}")
    print(f"OK tool_calls: {msg.get('tool_calls')}")
    print(f"usage: {resp.get('usage')}")
    return True


def main() -> int:
    args = parse_args()
    print_config(args)

    ok = chat_test("planner", args.router_api_base, args.router_api_key, args.planner_model, args.max_tokens)
    if args.router_model != args.planner_model:
        ok = chat_test("router", args.router_api_base, args.router_api_key, args.router_model, args.max_tokens) and ok

    if not args.skip_tools:
        ok = tools_test(args.router_api_base, args.router_api_key, args.planner_model, args.max_tokens) and ok

    if not args.skip_worker:
        if not args.worker_api_key:
            print("\n[worker] SKIPPED: QWEN_API_KEY or SILICONFLOW_API_KEY is not set.", file=sys.stderr)
            ok = False
        else:
            ok = chat_test("worker", args.worker_api_base, args.worker_api_key, args.worker_model, args.max_tokens) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
