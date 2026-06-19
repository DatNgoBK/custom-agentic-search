"""Custom Qdrant adapter for OpenViking.

Subclasses ``QdrantCollectionAdapter`` and adds production hardening:
pydantic config validation, tenacity retry + pybreaker circuit breaker,
Prometheus metrics, deterministic UUIDv5 point IDs, HNSW + int8
quantization tuning, and a ``health_check`` method.

Loaded via the dotted-path mechanism in
``openviking/storage/vectordb_adapters/factory.py:33-42`` — set
``backend`` to the full class path in ``ov.conf`` and OpenViking imports
this class via ``importlib``. No fork required.

See the README's "Integration approach" section for a longer write-up
of why we subclass instead of rewriting the upstream adapter.
"""
from __future__ import annotations

from typing import Any

from openviking.storage.expr import FilterExpr
from openviking.storage.vectordb_adapters.qdrant_adapter import QdrantCollectionAdapter

from rag_qdrant.adapters.config import CustomAdapterParams
from rag_qdrant.adapters.identity import deterministic_point_id
from rag_qdrant.adapters.resilience import make_breaker, run_resilient
from rag_qdrant.observability import get_logger

log = get_logger("rag_qdrant.adapter")


class CustomQdrantCollectionAdapter(QdrantCollectionAdapter):
    """Production-flavored Qdrant adapter for OpenViking.

    Inherits from upstream:
      * ``_compile_filter`` (300+ lines of filter DSL → Qdrant must/should)
      * ``_normalize_record_for_write/read`` (URI scope + parent_uri tracking)
      * ``_load_existing_collection_if_needed`` / ``_create_backend_collection``
      * Index meta scaffolding

    Overrides:
      * ``from_config`` — pydantic validation
      * ``_build_default_index_meta`` — HNSW + quantization injection
      * ``_normalize_record_for_write`` — deterministic UUID
      * I/O ops (upsert/query/get/delete/count) — wrapped with retry/breaker/metrics
    """

    mode: str = "qdrant_custom"

    # ------------------------------------------------------------------ ctor

    def __init__(
        self,
        *,
        params: CustomAdapterParams,
        project_name: str,
        collection_name: str,
        index_name: str,
        distance_metric: str,
    ) -> None:
        super().__init__(
            url=params.url,
            api_key=params.api_key,
            timeout_seconds=params.timeout_seconds,
            project_name=project_name,
            collection_name=collection_name,
            index_name=index_name,
            distance_metric=distance_metric,
            dense_vector_name=params.dense_vector_name,
            sparse_vector_name=params.sparse_vector_name,
            meta_collection_name=params.meta_collection_name,
            enable_text_index=params.enable_text_index,
        )
        self._params = params
        self._breaker = make_breaker(
            fail_max=params.breaker_fail_max,
            reset_timeout=params.breaker_reset_seconds,
            name=f"qdrant:{collection_name}",
        )
        log.info(
            "adapter.constructed",
            collection=collection_name,
            project=project_name,
            url=params.url,
            distance=distance_metric,
            quantization=params.quantization.model_dump(),
            hnsw=params.hnsw.model_dump(),
        )

    # ------------------------------------------------------------- factory

    @classmethod
    def from_config(cls, config: Any) -> CustomQdrantCollectionAdapter:
        try:
            params = CustomAdapterParams.from_config(config)
        except Exception as exc:
            raise ValueError(f"Invalid Qdrant adapter config: {exc}") from exc

        return cls(
            params=params,
            project_name=getattr(config, "project_name", None) or "default",
            collection_name=getattr(config, "name", None) or "context",
            index_name=getattr(config, "index_name", None) or "default",
            distance_metric=getattr(config, "distance_metric", None) or "cosine",
        )

    # ------------------------------------------------ index meta override

    def _build_default_index_meta(
        self,
        *,
        index_name: str,
        distance: str,
        use_sparse: bool,
        sparse_weight: float,
        scalar_index_fields: list[str],
    ) -> dict[str, Any]:
        """Inject HNSW + quantization tuning on top of the parent's meta."""
        meta = super()._build_default_index_meta(
            index_name=index_name,
            distance=distance,
            use_sparse=use_sparse,
            sparse_weight=sparse_weight,
            scalar_index_fields=scalar_index_fields,
        )
        vector_index = meta.setdefault("VectorIndex", {})

        vector_index["HNSW"] = {
            "M": self._params.hnsw.m,
            "EfConstruct": self._params.hnsw.ef_construct,
            "FullScanThreshold": self._params.hnsw.full_scan_threshold,
        }
        vector_index["IndexingThreshold"] = self._params.indexing_threshold

        if self._params.quantization.enabled:
            vector_index["Quantization"] = {
                "Type": self._params.quantization.type,
                "AlwaysRam": self._params.quantization.always_ram,
            }

        log.debug(
            "adapter.index_meta_built",
            index_name=index_name,
            distance=distance,
            use_sparse=use_sparse,
            quantized=self._params.quantization.enabled,
        )
        return meta

    # ------------------------------------------- record id determinism

    def _normalize_record_for_write(self, record: dict[str, Any]) -> dict[str, Any]:
        """Inject a deterministic UUID, then let the parent handle URI scope."""
        normalized = super()._normalize_record_for_write(record)
        if not normalized.get("id"):
            normalized["id"] = deterministic_point_id(normalized)
        return normalized

    # --------------------------------------------- instrumented I/O ops
    #
    # Each public op delegates to the parent adapter through ``run_resilient``
    # so retry, circuit breaker, and metrics apply uniformly. We use
    # ``_call_parent`` instead of inline ``super()`` because ``super()`` is
    # only magic inside method bodies — inside a lambda it loses its
    # implicit class arg and silently misbehaves.

    def _call_parent(self, method_name: str, /, *args: Any, **kwargs: Any) -> Any:
        """Invoke a method on QdrantCollectionAdapter (the parent class)."""
        parent_method = getattr(QdrantCollectionAdapter, method_name)
        return parent_method(self, *args, **kwargs)

    def upsert(self, data: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
        return run_resilient(self, "upsert", lambda: self._call_parent("upsert", data))

    def query(
        self,
        *,
        query_vector: list[float] | None = None,
        sparse_query_vector: dict[str, float] | None = None,
        filter: dict[str, Any] | FilterExpr | None = None,  # noqa: A002 - upstream signature
        limit: int = 10,
        offset: int = 0,
        output_fields: list[str] | None = None,
        order_by: str | None = None,
        order_desc: bool = False,
    ) -> list[dict[str, Any]]:
        return run_resilient(
            self,
            "query",
            lambda: self._call_parent(
                "query",
                query_vector=query_vector,
                sparse_query_vector=sparse_query_vector,
                filter=filter,
                limit=limit,
                offset=offset,
                output_fields=output_fields,
                order_by=order_by,
                order_desc=order_desc,
            ),
        )

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        return run_resilient(self, "get", lambda: self._call_parent("get", ids))

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        filter: dict[str, Any] | FilterExpr | None = None,  # noqa: A002
        limit: int = 100_000,
    ) -> int:
        return run_resilient(
            self,
            "delete",
            lambda: self._call_parent("delete", ids=ids, filter=filter, limit=limit),
        )

    def count(self, filter: dict[str, Any] | FilterExpr | None = None) -> int:  # noqa: A002
        return run_resilient(self, "count", lambda: self._call_parent("count", filter))

    # ----------------------------------------------------- health check

    def health_check(self) -> dict[str, Any]:
        """Structured health probe for ops dashboards / k8s readiness.

        Never raises; returns ``status: down`` on any error so a probe handler
        can render the response without try/except.
        """
        info: dict[str, Any] = {
            "status": "down",
            "collection": self.collection_name,
            "physical_collection": self.physical_collection_name,
            "url": self._params.url,
            "breaker": {
                "name": self._breaker.name,
                "state": self._breaker.current_state,
                "fail_counter": self._breaker.fail_counter,
            },
            "collection_exists": False,
            "vector_count": None,
            "error": None,
        }
        try:
            exists = self.collection_exists()
            info["collection_exists"] = bool(exists)
            if exists:
                info["vector_count"] = self.count(filter=None)
                info["status"] = "ok"
            else:
                info["status"] = "degraded"  # connected but collection not provisioned yet
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("adapter.health_check_failed", error=info["error"])
        return info
