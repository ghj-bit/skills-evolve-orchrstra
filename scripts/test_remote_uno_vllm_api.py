"""Smoke test for OpenRouter OpenAI-compatible chat completions."""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from configs import load_secret_env


load_secret_env()

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_ID = os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


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
        raise SystemExit("OPENROUTER_API_KEY is not set. Set it before running this script.")

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
