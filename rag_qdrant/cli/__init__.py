"""Console-script entry points declared in pyproject.toml."""
from rag_qdrant.cli.commands import (
    eval_cmd,
    health_cmd,
    ingest_cmd,
    query_cmd,
)

__all__ = ["ingest_cmd", "query_cmd", "eval_cmd", "health_cmd"]
