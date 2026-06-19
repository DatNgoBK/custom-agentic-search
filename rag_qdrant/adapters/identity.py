"""Deterministic point IDs for idempotent upserts.

Qdrant accepts UUID or unsigned int as point ID. We derive a UUIDv5 from a
content fingerprint so re-ingesting the same chunk replaces (not duplicates).
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

# Stable namespace for this adapter family. Don't change once data exists.
_NAMESPACE = uuid.UUID("4f1c0e0c-2b8e-5d6a-9f1c-0e0c2b8e5d6a")


def deterministic_point_id(record: dict[str, Any]) -> str:
    """Return a UUIDv5 derived from a record's identity-bearing fields.

    Priority: explicit `id` → uri+chunk_index → uri+content hash → fallback hash.
    """
    explicit = record.get("id")
    if isinstance(explicit, str) and explicit:
        return _coerce_uuid(explicit)

    uri = record.get("uri") or record.get("source") or ""
    chunk_index = record.get("chunk_index")
    text = record.get("text") or record.get("content") or ""

    if uri and chunk_index is not None:
        seed = f"{uri}::chunk={chunk_index}"
    elif uri and text:
        seed = f"{uri}::sha256={_sha256(text)}"
    else:
        seed = _sha256(text or "empty")

    return str(uuid.uuid5(_NAMESPACE, seed))


def _coerce_uuid(value: str) -> str:
    """If `value` is already a UUID, keep it; otherwise hash it into UUIDv5."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(_NAMESPACE, value))


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
