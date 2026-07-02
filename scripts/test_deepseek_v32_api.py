"""Minimal OpenAI-compatible API smoke test for DeepSeek."""

from __future__ import annotations

import os

from openai import OpenAI

from configs import load_secret_env


load_secret_env()

API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL_ID = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-v4-flash")


def _unset_missing_cert_envs() -> None:
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(key)
        if value and not os.path.isfile(value):
            print(f"Unset {key}: file not found ({value})")
            os.environ.pop(key, None)


def main() -> None:
    _unset_missing_cert_envs()
    if not API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Set it before running this script.")

    client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=60, max_retries=0)
    print(f"API base: {API_BASE}")
    print(f"Model: {MODEL_ID}")

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        temperature=0,
        max_tokens=16,
    )

    message = response.choices[0].message.content or ""
    print("Response:")
    print(message)
    if getattr(response, "usage", None):
        print("Usage:")
        print(response.usage)


if __name__ == "__main__":
    main()
