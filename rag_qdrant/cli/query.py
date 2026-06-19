"""Demonstrate agentic search against an ingested document.

Reads root_uri persisted by the ingest step and runs Vietnamese queries
through OpenViking's ``client.find()`` — every search hits Qdrant via the
CustomQdrantCollectionAdapter, so a green run proves the full chain works.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass

from rag_qdrant.cli.session import REPO_ROOT, open_viking_session
from rag_qdrant.observability import get_logger

STATE_FILE = REPO_ROOT / ".ingestion_state.json"
log = get_logger("rag_qdrant.cli.query")

DEMO_QUERIES: list[str] = [
    "Tổng tài sản của MSB cuối năm 2024 là bao nhiêu?",
    "Lợi nhuận trước thuế MSB năm 2024 đạt bao nhiêu tỷ đồng?",
    "Vốn điều lệ của MSB năm 2024 là bao nhiêu?",
    "Mức tăng trưởng tín dụng của MSB năm 2024?",
    "MSB chuyển đổi hệ thống ngân hàng lõi từ gì sang gì?",
    "Ai là Tổng giám đốc của MSB?",
    "Chiến lược phát triển bền vững ESG của MSB?",
    "Mã chứng khoán của MSB là gì?",
]


@dataclass
class Hit:
    uri: str
    score: float
    snippet: str


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  rag-query                                    # run 8 demo queries\n"
            '  rag-query --query "Lợi nhuận MSB 2024?"      # ask one custom question\n'
            "  rag-query -i                                 # interactive REPL mode\n"
        ),
    )
    p.add_argument("--limit", type=int, default=5, help="Top-K hits per query (default: 5)")
    p.add_argument(
        "--query",
        action="append",
        help="Ask a custom question (repeatable). Overrides the demo set.",
    )
    p.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Enter REPL: type a question, get answers, repeat. Ctrl+D / 'exit' to quit.",
    )
    p.add_argument("--snippet-chars", type=int, default=180)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])

    if not STATE_FILE.exists():
        print(f"::error:: state file not found at {STATE_FILE}", file=sys.stderr)
        print("  Run `make ingest` first.", file=sys.stderr)
        return 2

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    root_uri = state.get("root_uri")
    if not root_uri:
        print("::error:: state file is missing root_uri — re-run ingest", file=sys.stderr)
        return 2

    with open_viking_session() as client:
        _print_header(state, args.limit, root_uri)

        if args.interactive:
            return _run_interactive(client, root_uri, args)

        queries = list(args.query) if args.query else list(DEMO_QUERIES)
        summary = _run_batch(client, root_uri, queries, args)
        _print_summary(summary)
        return 0 if all(n > 0 for _, n, _, _ in summary) else 1


def _run_batch(
    client, root_uri: str, queries: list[str], args
) -> list[tuple[str, int, float | None, float]]:
    """Run a fixed list of queries and return summary rows."""
    summary: list[tuple[str, int, float | None, float]] = []
    for idx, q in enumerate(queries, start=1):
        print()
        print(f"\033[1m[{idx}/{len(queries)}] Q: {q}\033[0m")
        latency, hits = _run_one(client, root_uri, q, args)
        summary.append((q, len(hits), hits[0].score if hits else None, latency))
        _print_hits(hits, latency, args.snippet_chars)
    return summary


def _run_interactive(client, root_uri: str, args) -> int:
    """REPL: prompt the user for queries, print results, loop until quit."""
    print()
    print("\033[1mInteractive mode — type a question, press Enter.\033[0m")
    print("  • Empty line, 'exit', 'quit', or Ctrl+D to leave.")
    print("  • Vietnamese works best (the corpus is the MSB 2024 report).")
    print()

    while True:
        try:
            question = input("\033[1mQ›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit", ":q"}:
            break

        latency, hits = _run_one(client, root_uri, question, args)
        _print_hits(hits, latency, args.snippet_chars)
        print()

    print("Bye.")
    return 0


def _run_one(client, root_uri: str, question: str, args) -> tuple[float, list[Hit]]:
    """Execute a single query, return (latency_seconds, hits)."""
    t0 = time.monotonic()
    try:
        results = client.find(query=question, target_uri=root_uri, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 - surface to user, don't crash
        latency = time.monotonic() - t0
        print(f"  ::error:: {type(exc).__name__}: {exc}")
        return latency, []
    latency = time.monotonic() - t0
    return latency, _coerce_hits(results, snippet_chars=args.snippet_chars)


def _print_hits(hits: list[Hit], latency: float, snippet_chars: int) -> None:
    if not hits:
        print("  (no results)")
        return
    for r, hit in enumerate(hits, start=1):
        snippet = textwrap.shorten(
            hit.snippet.replace("\n", " "), width=snippet_chars, placeholder="…"
        )
        print(f"  {r}. [{hit.score:6.3f}] {hit.uri}")
        print(f"     {snippet}")
    print(f"  ⏱  {latency*1000:.0f} ms")


def _print_header(state: dict, limit: int, root_uri: str) -> None:
    print()
    print("=" * 80)
    print(" AGENTIC SEARCH DEMO — MSB Annual Report 2024 (Vietnamese)")
    print("=" * 80)
    print(f"  Root URI:          {root_uri}")
    print(f"  Qdrant collection: {state.get('qdrant_collection')}")
    print(f"  Embed model:       {state.get('embed_model')}  ({state.get('embed_dim')}d)")
    print("  Adapter:           rag_qdrant.adapters.custom_qdrant_adapter")
    print("                     .CustomQdrantCollectionAdapter")
    print(f"  Top-K per query:   {limit}")
    print("=" * 80)


def _print_summary(rows: list[tuple[str, int, float | None, float]]) -> None:
    print()
    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"  {'Query':<60} {'Hits':>5} {'Top':>7} {'Lat (ms)':>9}")
    print(f"  {'-'*60} {'-'*5} {'-'*7} {'-'*9}")
    for q, n, top, lat in rows:
        qshort = q if len(q) <= 58 else q[:55] + "..."
        top_str = f"{top:.3f}" if top is not None else " —"
        print(f"  {qshort:<60} {n:>5} {top_str:>7} {lat*1000:>9.0f}")

    ok = sum(1 for _, n, _, _ in rows if n > 0)
    print()
    print(f"  Result: {ok}/{len(rows)} queries returned at least one hit")


def _coerce_hits(results: object, *, snippet_chars: int) -> list[Hit]:
    """OpenViking find() returns FindResponse with a .resources list."""
    items = list(getattr(results, "resources", None) or [])
    if not items and isinstance(results, list):
        items = list(results)
    return [_to_hit(it, snippet_chars) for it in items]


def _to_hit(item: object, snippet_chars: int) -> Hit:
    uri = getattr(item, "uri", None) or "(no-uri)"
    score = getattr(item, "score", None) or 0.0
    content = (
        getattr(item, "content", None)
        or getattr(item, "abstract", None)
        or getattr(item, "snippet", None)
        or ""
    )
    return Hit(uri=str(uri), score=float(score), snippet=str(content)[: snippet_chars * 4])
