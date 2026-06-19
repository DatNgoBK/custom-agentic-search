#!/usr/bin/env python
"""Thin wrapper around rag_qdrant.cli.query — kept for `make query`.

Adds the repo root to sys.path so this works without `pip install -e .`
when invoked directly (e.g. by self-test scripts or `python scripts/03…`).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_qdrant.cli.query import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
