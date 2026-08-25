"""List the models your current provider actually offers.

    python list_models.py

Providers retire and rename models regularly, so a hardcoded default goes stale
and you get a confusing 404 ("model does not exist") that looks like a key or
access problem. This asks the provider directly using whatever LLM_PROVIDER /
key you already have set, so you pick from the live list instead of guessing.

Uses the official OpenAI client rather than urllib on purpose: several providers
sit behind Cloudflare, which rejects urllib's default user-agent with a 403 that
looks exactly like an auth error but isn't.

Then set the one you want:
    $env:LLM_MODEL = "<id from the list>"
"""

import os
import sys

from llm import _resolve

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    provider = os.environ.get("LLM_PROVIDER", "huggingface").lower()
    try:
        base_url, current, api_key = _resolve()
    except RuntimeError as exc:
        print(f"!! {exc}")
        return 1

    print(f"provider : {provider}")
    print(f"base_url : {base_url}")
    print(f"current  : {current}\n")

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=1)
    try:
        listing = client.models.list()
    except Exception as exc:
        msg = str(exc)
        print(f"!! {type(exc).__name__}: {msg[:400]}")
        if "1010" in msg or "browser" in msg.lower():
            print("\nCloudflare blocked the request — not an auth problem.")
        elif "401" in msg or "invalid_api_key" in msg:
            print(f"\nAuth failed: check the API key for provider {provider!r}.")
        elif provider == "ollama":
            print("\nIs Ollama running? Start it with:  ollama serve")
        return 1

    ids = sorted(m.id for m in listing.data)
    if not ids:
        print("(provider returned no models)")
        return 1

    print(f"{len(ids)} model(s) available:\n")
    for mid in ids:
        print(f"  {mid}{'   <-- currently set' if mid == current else ''}")

    if current not in ids:
        print(
            f"\n!! Your configured model '{current}' is NOT in this list — that's the "
            '404 you saw.\n   Pick one above:\n     $env:LLM_MODEL = "<id>"'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
