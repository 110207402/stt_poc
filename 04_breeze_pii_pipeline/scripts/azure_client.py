"""Azure OpenAI chat client helper for KGI Phase 3 PII detection.

Loads credentials from the project-root `.env` file.
Required env vars:
  AZURE_API_KEY
  AZURE_ENDPOINT
  AZURE_API_VERSION
  AZURE_CHAT_DEPLOYMENT  (deployment name for the chat model, e.g. gpt-4o-mini)

Usage:
    from azure_client import get_chat_client, CHAT_DEPLOYMENT
    client = get_chat_client()
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT, messages=[...], ...
    )
"""

from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val or val.startswith("your-"):
        raise RuntimeError(
            f"Missing or placeholder env var {name}. "
            f"Copy {PROJECT_ROOT}/.env.example to .env and fill in real values."
        )
    return val


def get_chat_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=_require("AZURE_API_KEY"),
        api_version=_require("AZURE_API_VERSION"),
        azure_endpoint=_require("AZURE_ENDPOINT"),
    )


CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")


if __name__ == "__main__":
    print("=== Azure config check ===")
    for v in ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_VERSION", "AZURE_CHAT_DEPLOYMENT"]:
        val = os.environ.get(v, "")
        if v == "AZURE_API_KEY":
            shown = val[:6] + "…" + val[-4:] if val else "(missing)"
        else:
            shown = val or "(missing)"
        print(f"  {v}: {shown}")

    print("\n=== Smoke test: list chat deployment via small completion ===")
    try:
        client = get_chat_client()
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": "say 'ok' in one word"}],
            max_tokens=5,
            temperature=0,
        )
        print(f"  deployment={CHAT_DEPLOYMENT}  reply={resp.choices[0].message.content!r}  "
              f"tokens={resp.usage.total_tokens}")
        print("  ✓ Azure chat OK")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
