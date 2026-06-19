"""Embedding client — thin httpx wrapper over OpenAI-compatible /v1/embeddings.

Works with OpenRouter, OpenAI direct, local TEI, or any drop-in replacement
that respects the OpenAI embeddings schema. We use ``httpx`` instead of the
``openai`` SDK so retry/error taxonomy stays under our control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag_qdrant.observability import EMBED_LATENCY, get_logger
from rag_qdrant.settings import EmbedSettings, get_settings

log = get_logger("rag_qdrant.embedding")

# Retry only for *transient* failures. Auth errors (401), bad request (400),
# and unsupported model (404) are programming errors → fail fast.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class EmbeddingError(RuntimeError):
    """Raised when the embedding endpoint cannot be reached or returns garbage."""


EmbedKind = Literal["passage", "query"]


@dataclass(frozen=True)
class EmbedRequest:
    """A batch to embed, tagged as passage (document) or query.

    Some models (e5, bge-m3, instructor) need ``passage:`` / ``query:``
    prefixes; OpenAI models don't. We auto-detect from the model name —
    the ``kind`` here keeps the call site honest about intent.
    """

    texts: list[str]
    kind: EmbedKind


class EmbeddingClient:
    """Synchronous embedding client. One instance per app is plenty."""

    def __init__(
        self,
        settings: EmbedSettings | None = None,
        *,
        max_batch_size: int = 64,
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self._settings = settings or get_settings().embed
        self._max_batch_size = max_batch_size
        self._timeout = timeout_seconds
        self._max_retries = max_retries

        self._needs_prefix = self._detect_prefix_requirement(self._settings.model)

        headers = {"Content-Type": "application/json"}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"

        self._client = httpx.Client(
            base_url=self._settings.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )
        log.info(
            "embedding_client.constructed",
            base_url=self._settings.base_url,
            model=self._settings.model,
            dim=self._settings.dim,
            needs_prefix=self._needs_prefix,
        )

    # ----------------------------------------------------------------- public

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed documents — call at ingestion time."""
        return self._embed_batched(EmbedRequest(texts=texts, kind="passage"))

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed search queries — call at retrieval time."""
        return self._embed_batched(EmbedRequest(texts=texts, kind="query"))

    def health_check(self) -> dict[str, Any]:
        """Quick liveness probe: embed a single token, verify shape.

        Never raises; returns ``{"status": "down", "error": ...}`` on any
        error so it's safe to call from a readiness handler.
        """
        try:
            vec = self.embed_queries(["ping"])[0]
            return {
                "status": "ok",
                "model": self._settings.model,
                "base_url": self._settings.base_url,
                "dim_observed": len(vec),
                "dim_expected": self._settings.dim,
                "dim_match": len(vec) == self._settings.dim,
            }
        except Exception as exc:
            return {
                "status": "down",
                "model": self._settings.model,
                "base_url": self._settings.base_url,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----------------------------------------------------------------- internal

    @staticmethod
    def _detect_prefix_requirement(model: str) -> bool:
        """Return True if the model expects ``passage:`` / ``query:`` prefixes."""
        lower = model.lower()
        return any(needle in lower for needle in ("e5", "bge-m3", "instructor"))

    def _apply_prefix(self, req: EmbedRequest) -> list[str]:
        if not self._needs_prefix:
            return req.texts
        prefix = "query: " if req.kind == "query" else "passage: "
        return [f"{prefix}{t}" for t in req.texts]

    def _embed_batched(self, req: EmbedRequest) -> list[list[float]]:
        if not req.texts:
            return []

        prepared = self._apply_prefix(req)
        out: list[list[float]] = []
        for i in range(0, len(prepared), self._max_batch_size):
            batch = prepared[i : i + self._max_batch_size]
            with EMBED_LATENCY.labels(kind=req.kind).time():
                out.extend(self._post_one_batch(batch))
        return out

    def _post_one_batch(self, batch: list[str]) -> list[list[float]]:
        retrying = Retrying(
            stop=stop_after_attempt(max(self._max_retries, 1)),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception_type(EmbeddingError),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._post_once(batch)
        raise EmbeddingError("retrying loop exited without yielding")  # pragma: no cover

    def _post_once(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self._settings.model, "input": batch}
        try:
            response = self._client.post("/embeddings", json=payload)
        except httpx.RequestError as exc:
            # Network error → transient → retry
            raise EmbeddingError(f"network error talking to embed endpoint: {exc}") from exc

        if response.status_code in _RETRYABLE_STATUS:
            raise EmbeddingError(
                f"transient HTTP {response.status_code} from embed endpoint: "
                f"{response.text[:200]}"
            )

        if response.status_code >= 400:
            # Permanent error: don't retry, surface to caller verbatim.
            raise RuntimeError(
                f"embed endpoint returned HTTP {response.status_code}: {response.text[:500]}"
            )

        data = response.json()
        try:
            vectors = [item["embedding"] for item in data["data"]]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"unexpected embed response shape: {data}") from exc

        if vectors and len(vectors[0]) != self._settings.dim:
            log.warning(
                "embedding_client.dim_mismatch",
                expected=self._settings.dim,
                observed=len(vectors[0]),
                model=self._settings.model,
            )
        return vectors
