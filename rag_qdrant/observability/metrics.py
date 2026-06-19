"""Prometheus metrics. Import counters/histograms from here."""
from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

QDRANT_OPERATIONS = Counter(
    "qdrant_operations_total",
    "Number of Qdrant operations performed.",
    labelnames=("operation", "collection"),
)

QDRANT_ERRORS = Counter(
    "qdrant_errors_total",
    "Number of Qdrant operations that raised an exception.",
    labelnames=("operation", "collection", "error_type"),
)

QDRANT_LATENCY = Histogram(
    "qdrant_operation_latency_seconds",
    "Latency of Qdrant operations.",
    labelnames=("operation", "collection"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

EMBED_LATENCY = Histogram(
    "embedding_request_latency_seconds",
    "Latency of embedding HTTP requests.",
    labelnames=("kind",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_pipeline_latency_seconds",
    "End-to-end retrieval pipeline latency by stage.",
    labelnames=("stage",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


def start_metrics_server(port: int) -> None:
    """Expose /metrics on the given port."""
    start_http_server(port)
