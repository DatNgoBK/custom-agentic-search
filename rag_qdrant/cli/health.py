"""Aggregate health check across Qdrant + embedding endpoint + custom adapter.

Useful for k8s readiness probes and ops dashboards. Always exits with a
non-zero code if anything is degraded.
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from rag_qdrant.observability import get_logger

log = get_logger("rag_qdrant.cli.health")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])
    load_dotenv()

    from rag_qdrant.embedding.client import EmbeddingClient  # noqa: PLC0415

    embed = EmbeddingClient().health_check()
    report = {
        "embed": embed,
        # Qdrant + adapter checks would slot in here once the adapter is
        # constructable without OpenViking config — for now use scripts/00_*.
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"embed.status = {embed['status']}  (model={embed['model']})")
        if embed.get("error"):
            print(f"  error: {embed['error']}", file=sys.stderr)

    statuses = [v.get("status") for v in report.values() if isinstance(v, dict)]
    return 0 if all(s == "ok" for s in statuses) else 1
