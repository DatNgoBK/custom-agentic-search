"""Resilience helper: run a callable with retry + circuit breaker + metrics.

Design: breaker wraps retry (not vice versa) so transient hiccups absorbed
by retry don't trip the breaker — only the *final* failure does. The
``exclude`` list on the breaker keeps programming bugs (TypeError,
ValueError, KeyError) from tripping the circuit on a single failure.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import pybreaker
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_qdrant.observability import QDRANT_ERRORS, QDRANT_LATENCY, QDRANT_OPERATIONS, get_logger

if TYPE_CHECKING:
    from rag_qdrant.adapters.custom_qdrant_adapter import CustomQdrantCollectionAdapter

T = TypeVar("T")
log = get_logger("rag_qdrant.adapter.resilience")

# Conservative: only retry transient I/O errors. Programming errors (Type/Value/
# Key/Lookup) must surface immediately so bugs don't get masked by retries.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class AdapterCircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open (Qdrant declared unhealthy)."""


def make_breaker(*, fail_max: int, reset_timeout: int, name: str) -> pybreaker.CircuitBreaker:
    return pybreaker.CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        name=name,
        exclude=[ValueError, TypeError, KeyError, LookupError],
    )


def run_resilient(
    adapter: CustomQdrantCollectionAdapter,
    operation: str,
    fn: Callable[[], T],
) -> T:
    """Execute ``fn`` with metrics, retry, and circuit breaker.

    All knobs come from ``adapter._params`` so per-instance tuning works.

    Raises:
        AdapterCircuitOpenError: if the breaker is open before/after the call.
        Exception: whatever fn raises after retries exhausted.
    """
    collection = adapter.collection_name
    params = adapter._params
    breaker = adapter._breaker

    QDRANT_OPERATIONS.labels(operation=operation, collection=collection).inc()

    retrying = Retrying(
        stop=stop_after_attempt(max(params.max_retries, 1)),
        wait=wait_exponential_jitter(
            initial=params.retry_min_wait_seconds,
            max=params.retry_max_wait_seconds,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )

    def _retry_then_call() -> T:
        # tenacity ≥8: prefer iteration over deprecated __call__.
        for attempt in retrying:
            with attempt:
                return fn()
        # Unreachable: with reraise=True, the final exception propagates out
        # of the for-loop on the failing attempt.
        raise RuntimeError(  # pragma: no cover
            f"tenacity Retrying exited without yielding a result for {operation}"
        )

    try:
        with QDRANT_LATENCY.labels(operation=operation, collection=collection).time():
            return breaker.call(_retry_then_call)
    except pybreaker.CircuitBreakerError as exc:
        QDRANT_ERRORS.labels(
            operation=operation, collection=collection, error_type="circuit_open"
        ).inc()
        log.warning(
            "qdrant.circuit_open",
            operation=operation,
            collection=collection,
            breaker=breaker.name,
        )
        raise AdapterCircuitOpenError(
            f"Qdrant circuit '{breaker.name}' is open: {exc}"
        ) from exc
    except Exception as exc:
        QDRANT_ERRORS.labels(
            operation=operation,
            collection=collection,
            error_type=type(exc).__name__,
        ).inc()
        log.error(
            "qdrant.operation_failed",
            operation=operation,
            collection=collection,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
