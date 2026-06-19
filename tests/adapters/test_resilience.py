"""Tests for retry + circuit breaker behavior."""
from __future__ import annotations

import pybreaker
import pytest

from rag_qdrant.adapters.custom_qdrant_adapter import CustomQdrantCollectionAdapter
from rag_qdrant.adapters.resilience import (
    AdapterCircuitOpenError,
    make_breaker,
    run_resilient,
)
from tests.adapters.conftest import make_ov_config


def test_transient_error_is_retried_then_succeeds(adapter):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("temporary")
        return "ok"

    assert run_resilient(adapter, "upsert", flaky) == "ok"
    assert calls["n"] == 2


def test_programming_error_is_not_retried(adapter):
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("bad arg")

    with pytest.raises(ValueError):
        run_resilient(adapter, "upsert", boom)
    assert calls["n"] == 1


def test_breaker_opens_after_repeated_failure():
    """With fail_max=2 and retry disabled, the second failure opens the breaker.

    pybreaker counts each failure once retries are exhausted:
        call 1 → ConnectionError       (counter = 1)
        call 2 → AdapterCircuitOpenError (counter = 2 → trip → wraps the error)
        call 3 → AdapterCircuitOpenError (breaker open, short-circuits)
    """
    adapter = CustomQdrantCollectionAdapter.from_config(
        make_ov_config(custom_params={
            "url": "http://localhost:6333",
            "max_retries": 1,
            "breaker_fail_max": 2,
            "breaker_reset_seconds": 30,
        })
    )

    def always_fail():
        raise ConnectionError("nope")

    with pytest.raises(ConnectionError):
        run_resilient(adapter, "upsert", always_fail)

    with pytest.raises(AdapterCircuitOpenError):
        run_resilient(adapter, "upsert", always_fail)

    with pytest.raises(AdapterCircuitOpenError):
        run_resilient(adapter, "upsert", always_fail)


def test_make_breaker_excludes_programming_errors():
    breaker = make_breaker(fail_max=1, reset_timeout=10, name="t")

    for _ in range(5):
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("x")))
    assert breaker.current_state == pybreaker.STATE_CLOSED
