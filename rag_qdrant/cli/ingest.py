"""Ingest a markdown document into Qdrant via OpenViking + custom adapter.

Idempotent: deterministic UUIDv5 point IDs in the adapter mean re-running
this with the same markdown replaces — never duplicates — existing vectors.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag_qdrant.cli.session import REPO_ROOT, open_viking_session
from rag_qdrant.observability import get_logger
from rag_qdrant.preprocessing import chunk_markdown, clean_marker_output
from rag_qdrant.settings import get_settings

STATE_FILE = REPO_ROOT / ".ingestion_state.json"
DEFAULT_MARKDOWN = REPO_ROOT / "output" / "source" / "source.md"
log = get_logger("rag_qdrant.cli.ingest")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("markdown", nargs="?", default=str(DEFAULT_MARKDOWN))
    parser.add_argument("--ov-conf", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or [])

    md_path = Path(args.markdown).expanduser().resolve()
    if not md_path.exists():
        log.error("ingest.markdown_missing", path=str(md_path))
        return 2

    # --- PREPROCESSING ---
    log.info("ingest.preprocessing_start", source=str(md_path))
    cleaned_path = md_path.parent / f"{md_path.stem}_cleaned.md"
    chunks_dir = md_path.parent / "chunks"

    # Load optional OCR patches for this specific PDF dataset
    patches_path = REPO_ROOT / "rag_qdrant" / "preprocessing" / "msb_patches.json"
    patches = {}
    if patches_path.exists():
        patches = json.loads(patches_path.read_text(encoding="utf-8"))
        log.info("ingest.loaded_patches", count=len(patches))

    clean_stats = clean_marker_output(md_path, cleaned_path, patches=patches)
    log.info("ingest.cleaned", **clean_stats)

    chunk_stats = chunk_markdown(cleaned_path, chunks_dir)
    log.info("ingest.chunked", **chunk_stats)
    # ---------------------

    ov_conf_path = Path(args.ov_conf).expanduser().resolve() if args.ov_conf else None

    settings = get_settings()
    start = time.monotonic()

    with open_viking_session(ov_conf=ov_conf_path) as client:
        log.info("ingest.add_resource_start", source=str(chunks_dir), wait=not args.no_wait)
        result = client.add_resource(
            path=str(chunks_dir),
            wait=not args.no_wait,
            timeout=args.timeout if not args.no_wait else None,
            build_index=True,
            summarize=False,
        )
        elapsed = round(time.monotonic() - start, 2)
        log.info("ingest.add_resource_done", elapsed_s=elapsed)

        root_uri = result.get("root_uri") or result.get("uri") or result.get("path")

        STATE_FILE.write_text(
            json.dumps(
                {
                    "markdown": str(md_path),
                    "cleaned": str(cleaned_path),
                    "chunks_dir": str(chunks_dir),
                    "clean_stats": clean_stats,
                    "chunk_stats": chunk_stats,
                    "root_uri": root_uri,
                    "ingest_result": _jsonable(result),
                    "elapsed_seconds": elapsed,
                    "qdrant_collection": settings.qdrant.collection,
                    "embed_model": settings.embed.model,
                    "embed_dim": settings.embed.dim,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("✅ Ingestion complete")
        print(f"   markdown:      {md_path}")
        print(f"   root_uri:      {root_uri}")
        print(f"   elapsed:       {elapsed}s")
        print(f"   state file:    {STATE_FILE}")
        print(f"   Qdrant coll:   {settings.qdrant.collection}")
    return 0


def _jsonable(value: object) -> object:
    """Best-effort coerce arbitrary OpenViking response objects into JSON."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
        return str(value)
