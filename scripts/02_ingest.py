#!/usr/bin/env python
"""Thin wrapper around rag_qdrant.cli.ingest — kept for `make ingest`.

Logic lives in the package so it's reachable as both a console script
(`rag-ingest`) and a module (`python -m rag_qdrant.cli.ingest`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_qdrant.cli.ingest import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
