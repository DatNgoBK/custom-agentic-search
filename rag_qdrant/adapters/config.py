"""Pydantic config models for the custom Qdrant adapter.

Wraps OpenViking's loose config dict (duck-typed via getattr) in a strict
schema so misconfiguration fails at adapter construction with a clear message
instead of crashing somewhere deep in Qdrant client code.
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

# HNSW tuning — see https://qdrant.tech/documentation/concepts/indexing/#hnsw-index
DEFAULT_HNSW_M = 32           # connections per layer; higher → better recall, more memory
DEFAULT_HNSW_EF_CONSTRUCT = 256  # ef during build; quality knob
DEFAULT_INDEXING_THRESHOLD = 20000   # vectors before HNSW kicks in (small dataset → flat is fine)


class HNSWConfig(BaseModel):
    m: int = Field(default=DEFAULT_HNSW_M, ge=4, le=128)
    ef_construct: int = Field(default=DEFAULT_HNSW_EF_CONSTRUCT, ge=10, le=2048)
    full_scan_threshold: int = Field(default=10_000, ge=0)


class QuantizationConfig(BaseModel):
    """Scalar int8 quantization — typically -75% RAM with ~1% recall loss."""

    enabled: bool = True
    type: str = Field(default="int8")
    always_ram: bool = False  # if True, keep quantized vectors in RAM (faster but uses RAM)


class CustomAdapterParams(BaseModel):
    """Validated subset of `custom_params` from ov.conf for this adapter."""

    # Connection
    url: str = Field(..., description="Qdrant URL, e.g. http://localhost:6333")
    api_key: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    prefer_grpc: bool = Field(default=False, description="gRPC requires Qdrant on :6334")

    # Index tuning
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    indexing_threshold: int = Field(default=DEFAULT_INDEXING_THRESHOLD, ge=0)

    # Retry / circuit breaker
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_min_wait_seconds: float = Field(default=0.5, gt=0)
    retry_max_wait_seconds: float = Field(default=8.0, gt=0)
    breaker_fail_max: int = Field(default=5, ge=1)
    breaker_reset_seconds: int = Field(default=60, ge=1)

    # Naming
    dense_vector_name: str = Field(default="vector")
    sparse_vector_name: str = Field(default="sparse_vector")
    meta_collection_name: str = Field(default="__openviking_meta")
    enable_text_index: bool = Field(default=True)

    @field_validator("url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        if not v:
            raise ValueError("url must be a non-empty string")
        return v.strip().rstrip("/")

    # Fields we look up across every config layer in the same order.
    # ClassVar tells pydantic this is metadata, not a model field.
    _LAYERED_FIELDS: ClassVar[tuple[str, ...]] = (
        "url",
        "api_key",
        "timeout_seconds",
        "prefer_grpc",
        "max_retries",
        "retry_min_wait_seconds",
        "retry_max_wait_seconds",
        "breaker_fail_max",
        "breaker_reset_seconds",
        "dense_vector_name",
        "sparse_vector_name",
        "meta_collection_name",
        "enable_text_index",
        "indexing_threshold",
    )

    @classmethod
    def from_config(cls, ov_config: Any) -> CustomAdapterParams:
        """Build params from OpenViking's VectorDBBackendConfig.

        Precedence (first non-None wins):
            1. ``ov_config.qdrant.<field>``         (typed Qdrant section)
            2. ``ov_config.<field>``                (top-level fallback)
            3. ``ov_config.custom_params[<field>]`` (loose dict, our extension)
        """
        qdrant_section = getattr(ov_config, "qdrant", None)
        custom_params = dict(getattr(ov_config, "custom_params", {}) or {})

        nested_params: dict[str, Any] = {}
        for key in cls._LAYERED_FIELDS:
            value = cls._lookup(key, qdrant_section, ov_config, custom_params)
            if value is not None:
                nested_params[key] = value

        # Nested objects only come from custom_params (Qdrant section is flat).
        for nested_key in ("hnsw", "quantization"):
            value = custom_params.get(nested_key)
            if isinstance(value, dict):
                nested_params[nested_key] = value

        if "url" not in nested_params:
            raise ValueError(
                "Custom Qdrant adapter requires a URL. Set one of: "
                "ov.conf 'storage.vectordb.qdrant.url', 'storage.vectordb.url', "
                "or 'storage.vectordb.custom_params.url'."
            )

        return cls.model_validate(nested_params)

    @staticmethod
    def _lookup(key: str, qdrant_section: Any, ov_config: Any, custom_params: dict) -> Any:
        """Walk the precedence chain and return the first non-None value."""
        if qdrant_section is not None:
            value = getattr(qdrant_section, key, None)
            if value is not None:
                return value

        value = getattr(ov_config, key, None)
        if value is not None:
            return value

        if key in custom_params:
            return custom_params[key]

        return None
