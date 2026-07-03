"""Smoke test for OpenRouter OpenAI-compatible chat completions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from openai import OpenAI

from configs import load_secret_env


load_secret_env()

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL_ID = os.environ.get("OPENROUTER_MODEL", "Qwen/Qwen3-8B")
API_KEY = (
    os.environ.get("QWEN_API_KEY")
    or os.environ.get("SILICONFLOW_API_KEY")
    or ""
)


def clear_invalid_cert_env() -> None:
    """Remove broken certificate env vars before httpx/OpenAI client init."""
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(name)
        if value and not Path(value).is_file():
            print(f"Unset {name}: file not found ({value})")
            os.environ.pop(name, None)


def main() -> None:
    clear_invalid_cert_env()
    if not API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY, QWEN_API_KEY, or SILICONFLOW_API_KEY must be set."
        )

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?",
            }
        ],
        extra_body={"reasoning": {"enabled": True}},
    )

    first_message = response.choices[0].message
    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
        {
            "role": "assistant",
            "content": first_message.content,
            "reasoning_details": getattr(first_message, "reasoning_details", None),
        },
        {"role": "user", "content": "Are you sure? Think carefully."},
    ]

    response2 = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        extra_body={"reasoning": {"enabled": True}},
    )

    message = response2.choices[0].message.content or ""
    print(f"API base: {BASE_URL}")
    print(f"Model: {MODEL_ID}")
    print("Response:")
    print(message)
    print()
    print("Usage:")
    print(response2.usage)


if __name__ == "__main__":
    main()
