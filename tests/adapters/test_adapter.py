"""Tests for the adapter itself: index meta, record normalization, health, factory."""
from __future__ import annotations

import importlib

from rag_qdrant.adapters.custom_qdrant_adapter import CustomQdrantCollectionAdapter
from tests.adapters.conftest import make_ov_config

# ---------------------------------------------------- index meta shape


def test_index_meta_includes_hnsw_and_quantization(adapter):
    meta = adapter._build_default_index_meta(
        index_name="default",
        distance="cosine",
        use_sparse=True,
        sparse_weight=0.3,
        scalar_index_fields=["uri", "name"],
    )
    vi = meta["VectorIndex"]
    assert vi["HNSW"]["M"] == 32
    assert vi["HNSW"]["EfConstruct"] == 256
    assert vi["Quantization"]["Type"] == "int8"
    assert vi.get("EnableSparse") is True
    assert "ScalarIndex" in meta


def test_index_meta_omits_quantization_when_disabled():
    cfg = make_ov_config(
        custom_params={"url": "http://localhost:6333", "quantization": {"enabled": False}}
    )
    a = CustomQdrantCollectionAdapter.from_config(cfg)
    meta = a._build_default_index_meta(
        index_name="default",
        distance="cosine",
        use_sparse=False,
        sparse_weight=0.0,
        scalar_index_fields=[],
    )
    assert "Quantization" not in meta["VectorIndex"]


# ------------------------------------------------ record id injection


def test_normalize_record_for_write_injects_uuid_when_missing(adapter):
    out = adapter._normalize_record_for_write(
        {"uri": "viking://doc/x", "chunk_index": 2, "text": "t"}
    )
    assert "id" in out
    assert len(out["id"]) == 36


def test_normalize_record_for_write_keeps_existing_id(adapter):
    explicit = "11111111-2222-3333-4444-555555555555"
    out = adapter._normalize_record_for_write(
        {"id": explicit, "uri": "viking://doc/x", "text": "t"}
    )
    assert out["id"] == explicit


# -------------------------------------------------------- health check


def test_health_check_returns_status_dict_without_raising(adapter, monkeypatch):
    def boom():
        raise ConnectionError("qdrant unreachable")

    monkeypatch.setattr(adapter, "collection_exists", boom)

    info = adapter.health_check()
    assert info["status"] == "down"
    assert "ConnectionError" in info["error"]
    assert "breaker" in info


def test_health_check_status_ok_when_collection_exists(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "collection_exists", lambda: True)
    monkeypatch.setattr(adapter, "count", lambda filter=None: 42)

    info = adapter.health_check()
    assert info["status"] == "ok"
    assert info["vector_count"] == 42


# --------------------------------------------- factory dotted-path wiring


def test_dotted_path_resolves_to_our_adapter():
    """The factory's importlib fallback should yield our class."""
    from openviking.storage.vectordb_adapters.factory import _ADAPTER_REGISTRY

    assert CustomQdrantCollectionAdapter not in _ADAPTER_REGISTRY.values()

    module = importlib.import_module("rag_qdrant.adapters.custom_qdrant_adapter")
    cls = module.CustomQdrantCollectionAdapter
    assert cls is CustomQdrantCollectionAdapter
