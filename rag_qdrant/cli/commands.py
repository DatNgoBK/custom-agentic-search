"""Console-script entry points: rag-ingest, rag-query, rag-eval, rag-health.

Each function is registered in pyproject.toml under [project.scripts]
and runs ``sys.exit(...)`` so shell exit codes are meaningful.
"""
from __future__ import annotations

import sys


def ingest_cmd() -> None:
    from rag_qdrant.cli.ingest import main  # noqa: PLC0415

    sys.exit(main(sys.argv[1:]))


def query_cmd() -> None:
    from rag_qdrant.cli.query import main  # noqa: PLC0415

    sys.exit(main(sys.argv[1:]))


def eval_cmd() -> None:
    print(
        "rag-eval: not yet implemented. "
        "See tests/eval/queries.yaml (template).",
        file=sys.stderr,
    )
    sys.exit(2)


def health_cmd() -> None:
    from rag_qdrant.cli.health import main  # noqa: PLC0415

    sys.exit(main(sys.argv[1:]))
