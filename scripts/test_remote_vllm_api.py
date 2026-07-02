#!/usr/bin/env python
"""Smoke test a remote vLLM OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import os
import sys
from openai import OpenAI


DEFAULT_API_BASE = (
    "https://nat-notebook-inspire.sii.edu.cn/"
    "ws-6040202d-b785-4b37-98b0-c68d65dd52ce/"
    "project-b795c114-135a-40db-b3d0-19b60f25237b/"
    "user-543feed4-0be2-4972-8987-a324af06c93f/"
    "vscode/38307e4a-ec50-46a5-bc8e-7328f9a72356/"
    "ec5560b0-34a3-4da6-ac4d-09fbd1c3c683/"
    "proxy/8000/v1"
)
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=os.environ.get("API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--prompt", default="请用一句话说明你是谁。")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    print(f"API base: {args.api_base}")
    print(f"Model: {args.model}")
    print("Sending chat completion request...")

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": args.prompt},
            ],
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    choice = resp.choices[0]
    print("\nResponse:")
    print(choice.message.content)
    print("\nUsage:")
    print(resp.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
