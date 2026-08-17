#!/usr/bin/env python3
"""CLI: regenerate eval/model_prices.json from OpenRouter's live catalog
(https://openrouter.ai/api/v1/models + per-model /endpoints), so
src/bixiascribe/pricing.py's cost estimates don't quietly go stale.

For each model id already present in eval/model_prices.json (plus any extra
ids passed via --model), picks the cheapest endpoint that reports
`supports_tools=True` unless --provider pins one explicitly -- picking a
tools=False endpoint would silently break wuxia_corpus_search the same way
an unpinned real run could (see CLAUDE.md's dialogue-agent tool-calling
gotcha, and eval/model_variants.json's notes on this exact failure mode for
z-ai/glm-5.2 / tencent/hy3). Network-only script (no crewai/litellm import,
matching pricing.py's own zero-dependency stance) -- offline entirely
otherwise.

Usage:
    python scripts/refresh_prices.py                       # refresh every existing entry
    python scripts/refresh_prices.py --model deepseek/deepseek-v4-flash-0731
    python scripts/refresh_prices.py --model z-ai/glm-5.2 --provider Novita
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bixiascribe.pricing import DEFAULT_PRICES_FILE  # noqa: E402

API_BASE = "https://openrouter.ai/api/v1"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 -- fixed https host
        return json.load(resp)


def _bare_id(model_id: str) -> str:
    """"openrouter/deepseek/deepseek-chat" -> "deepseek/deepseek-chat" -- this
    repo's model ids carry the litellm "openrouter/" prefix everywhere
    (llm.py/eval/model_variants.json), but OpenRouter's own API expects the
    bare id."""
    return model_id[len("openrouter/") :] if model_id.startswith("openrouter/") else model_id


def refresh_one(bare_id: str, *, provider: str | None, fetched_at: str) -> dict | None:
    """Fetch pricing for one bare model id. Returns None (with a stderr
    warning) if the model or the requested/cheapest-tools endpoint can't be
    found -- a caller should keep the existing entry rather than delete it
    on a transient lookup failure."""
    try:
        endpoints = _fetch_json(f"{API_BASE}/models/{bare_id}/endpoints")["data"]["endpoints"]
    except Exception as exc:  # noqa: BLE001 -- report and move on, don't crash the batch
        print(f"  ! {bare_id}: fetch failed ({exc})", file=sys.stderr)
        return None

    candidates = endpoints
    if provider:
        candidates = [e for e in endpoints if e["provider_name"] == provider]
        if not candidates:
            print(f"  ! {bare_id}: no endpoint from provider {provider!r}", file=sys.stderr)
            return None
    else:
        tool_capable = [
            e for e in endpoints if "tools" in (e.get("supported_parameters") or [])
        ]
        candidates = tool_capable or endpoints
        if not tool_capable:
            print(f"  ! {bare_id}: NO endpoint reports tools support", file=sys.stderr)

    chosen = min(
        candidates,
        key=lambda e: float(e["pricing"]["prompt"]) + float(e["pricing"]["completion"]),
    )
    p = chosen["pricing"]
    return {
        "prompt_usd_per_1m": round(float(p["prompt"]) * 1_000_000, 4),
        "completion_usd_per_1m": round(float(p["completion"]) * 1_000_000, 4),
        "provider": chosen["provider_name"],
        "context_length": chosen.get("context_length"),
        "max_completion_tokens": chosen.get("max_completion_tokens"),
        "supports_tools": "tools" in (chosen.get("supported_parameters") or []),
        "fetched_at": fetched_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prices-file",
        type=Path,
        default=DEFAULT_PRICES_FILE,
        help="eval/model_prices.json path.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Bare or 'openrouter/'-prefixed model id to add/refresh (repeatable). "
        "Default: refresh every model id already in --prices-file.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Pin to this OpenRouter provider slug instead of auto-picking the "
        "cheapest tools-capable endpoint. Only meaningful with a single --model.",
    )
    args = parser.parse_args()

    existing: dict = {}
    if args.prices_file.exists():
        existing = json.loads(args.prices_file.read_text(encoding="utf-8"))

    model_ids = args.model or [k for k in existing if not k.startswith("_")]
    if not model_ids:
        print("No model ids to refresh -- pass --model at least once.", file=sys.stderr)
        sys.exit(1)
    if args.provider and len(model_ids) > 1:
        print("--provider only makes sense with a single --model.", file=sys.stderr)
        sys.exit(1)

    fetched_at = date.today().isoformat()

    updated = 0
    for model_id in model_ids:
        bare = _bare_id(model_id)
        key = model_id if model_id.startswith("openrouter/") else f"openrouter/{bare}"
        print(f"Fetching {bare} ...")
        entry = refresh_one(bare, provider=args.provider, fetched_at=fetched_at)
        if entry is None:
            continue
        existing[key] = entry
        updated += 1
        print(
            f"  -> {key}: prompt=${entry['prompt_usd_per_1m']}/1M  "
            f"completion=${entry['completion_usd_per_1m']}/1M  provider={entry['provider']!r}  "
            f"tools={entry['supports_tools']}"
        )

    args.prices_file.parent.mkdir(parents=True, exist_ok=True)
    args.prices_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Updated {updated}/{len(model_ids)} entr(y/ies) in {args.prices_file}")


if __name__ == "__main__":
    main()
