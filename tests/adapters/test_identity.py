"""Tests for deterministic UUIDv5 point IDs."""
from __future__ import annotations

from rag_qdrant.adapters.identity import deterministic_point_id


def test_same_input_yields_same_id():
    rec = {"uri": "viking://doc/a", "chunk_index": 3, "text": "hello"}
    assert deterministic_point_id(rec) == deterministic_point_id(rec)


def test_different_chunks_yield_different_ids():
    a = deterministic_point_id({"uri": "viking://doc/a", "chunk_index": 0, "text": "x"})
    b = deterministic_point_id({"uri": "viking://doc/a", "chunk_index": 1, "text": "x"})
    assert a != b


def test_explicit_uuid_is_preserved():
    rec = {"id": "11111111-2222-3333-4444-555555555555", "text": "x"}
    assert deterministic_point_id(rec) == "11111111-2222-3333-4444-555555555555"


def test_non_uuid_explicit_id_is_coerced_to_uuidv5():
    rec = {"id": "not-a-uuid", "text": "x"}
    out = deterministic_point_id(rec)
    assert len(out) == 36 and out.count("-") == 4
