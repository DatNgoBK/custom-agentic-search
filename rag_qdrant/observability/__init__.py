"""Observability: structured logging, Prometheus metrics, optional OpenTelemetry."""
from rag_qdrant.observability.logging import configure_logging, get_logger
from rag_qdrant.observability.metrics import (
    EMBED_LATENCY,
    QDRANT_ERRORS,
    QDRANT_LATENCY,
    QDRANT_OPERATIONS,
    RETRIEVAL_LATENCY,
    start_metrics_server,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "EMBED_LATENCY",
    "QDRANT_ERRORS",
    "QDRANT_LATENCY",
    "QDRANT_OPERATIONS",
    "RETRIEVAL_LATENCY",
    "start_metrics_server",
]
