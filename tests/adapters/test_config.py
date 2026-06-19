"""Tests for CustomAdapterParams pydantic schema and config layering."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_qdrant.adapters.config import CustomAdapterParams
from tests.adapters.conftest import make_ov_config


def test_url_is_required():
    cfg = make_ov_config(url=None, qdrant=SimpleNamespace(url=None), custom_params={})
    with pytest.raises(ValueError, match="requires a URL"):
        CustomAdapterParams.from_config(cfg)


def test_trailing_slash_is_stripped():
    cfg = make_ov_config(url="http://localhost:6333/")
    p = CustomAdapterParams.from_config(cfg)
    assert p.url == "http://localhost:6333"


def test_falls_back_to_custom_params_when_qdrant_section_missing():
    cfg = SimpleNamespace(
        custom_params={"url": "http://qd:6333", "api_key": "from_custom"},
        qdrant=None,
    )
    p = CustomAdapterParams.from_config(cfg)
    assert p.url == "http://qd:6333"
    assert p.api_key == "from_custom"


def test_hnsw_and_quantization_defaults():
    p = CustomAdapterParams.from_config(make_ov_config())
    assert p.hnsw.m == 32
    assert p.hnsw.ef_construct == 256
    assert p.quantization.enabled is True
    assert p.quantization.type == "int8"


def test_invalid_hnsw_m_is_rejected():
    cfg = make_ov_config(custom_params={"url": "http://x:6333", "hnsw": {"m": 999}})
    with pytest.raises(ValueError):
        CustomAdapterParams.from_config(cfg)
