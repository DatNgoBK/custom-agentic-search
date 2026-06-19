"""Shared fixtures for the adapter test suite — no Qdrant, no network."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from rag_qdrant.adapters.custom_qdrant_adapter import CustomQdrantCollectionAdapter


def make_ov_config(**overrides: Any) -> SimpleNamespace:
    """Build a duck-typed OpenViking VectorDBBackendConfig stand-in."""
    base: dict[str, Any] = {
        "name": "test_collection",
        "project_name": "test_project",
        "index_name": "default",
        "distance_metric": "cosine",
        "backend": "rag_qdrant.adapters.custom_qdrant_adapter.CustomQdrantCollectionAdapter",
        "url": "http://localhost:6333",
        "qdrant": SimpleNamespace(url="http://localhost:6333", api_key="key", timeout_seconds=10),
        "custom_params": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def ov_config() -> SimpleNamespace:
    """Default OpenViking config stand-in. Override fields per test as needed."""
    return make_ov_config()


@pytest.fixture
def adapter() -> CustomQdrantCollectionAdapter:
    """Adapter instance built from the default config; never opens a connection."""
    return CustomQdrantCollectionAdapter.from_config(make_ov_config())
