"""Unit tests for the embedding client. No network calls.

Real-network tests live behind a marker and run only when an embedding
endpoint is configured + reachable (see tests/test_embedding_integration.py).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag_qdrant.embedding.client import EmbeddingClient, EmbeddingError
from rag_qdrant.settings import EmbedSettings


def _settings(**overrides) -> EmbedSettings:
    base = {
        "base_url": "http://embed.local/v1",
        "model": "openai/text-embedding-3-small",
        "dim": 1536,
        "api_key": "sk-test",
    }
    base.update(overrides)
    # bypass env loading by passing kwargs explicitly
    return EmbedSettings.model_construct(**base)


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


def _ok_payload(n: int, dim: int) -> dict:
    return {"data": [{"embedding": [0.1] * dim, "index": i} for i in range(n)]}


# ---------------------------------------------------------------- prefix logic


def test_e5_model_gets_passage_prefix():
    client = EmbeddingClient(settings=_settings(model="intfloat/multilingual-e5-base"))
    assert client._needs_prefix is True
    out = client._apply_prefix(MagicMock(texts=["abc"], kind="passage"))
    assert out == ["passage: abc"]


def test_openai_model_no_prefix():
    client = EmbeddingClient(settings=_settings(model="openai/text-embedding-3-small"))
    assert client._needs_prefix is False
    out = client._apply_prefix(MagicMock(texts=["abc"], kind="passage"))
    assert out == ["abc"]


def test_query_kind_uses_query_prefix():
    client = EmbeddingClient(settings=_settings(model="intfloat/multilingual-e5-base"))
    out = client._apply_prefix(MagicMock(texts=["x"], kind="query"))
    assert out == ["query: x"]


# -------------------------------------------------------------------- batching


def test_batch_split_when_input_exceeds_max_batch_size():
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_batch_size=2)

    posts: list[list[str]] = []

    def fake_post(_path: str, json: dict) -> MagicMock:
        posts.append(json["input"])
        return _mock_response(_ok_payload(len(json["input"]), settings.dim))

    with patch.object(client._client, "post", side_effect=fake_post):
        out = client.embed_passages(["a", "b", "c", "d", "e"])

    # 5 inputs, batch=2 → splits into [2, 2, 1]
    assert [len(p) for p in posts] == [2, 2, 1]
    assert len(out) == 5


def test_empty_input_returns_empty_list_no_request():
    client = EmbeddingClient(settings=_settings())
    with patch.object(client._client, "post") as mock_post:
        assert client.embed_queries([]) == []
        mock_post.assert_not_called()


# ------------------------------------------------------------------ retries


def test_retries_on_429_then_succeeds():
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_retries=3)

    responses = [
        _mock_response({"error": "rate limit"}, status=429),
        _mock_response(_ok_payload(1, settings.dim)),
    ]
    with patch.object(client._client, "post", side_effect=responses):
        out = client.embed_queries(["hi"])
    assert len(out) == 1


def test_no_retry_on_400_bad_request():
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_retries=3)

    bad = _mock_response({"error": "bad input"}, status=400)
    bad.text = '{"error": "bad input"}'
    with patch.object(client._client, "post", return_value=bad), pytest.raises(RuntimeError, match="HTTP 400"):
        client.embed_queries(["x"])


def test_no_retry_on_401_auth():
    """401 is permanent in this codebase — fail fast so the user sees a clear msg."""
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_retries=3)

    bad = _mock_response({"error": {"message": "missing auth"}}, status=401)
    bad.text = "missing auth"
    with patch.object(client._client, "post", return_value=bad), pytest.raises(RuntimeError, match="HTTP 401"):
        client.embed_queries(["x"])


def test_retry_then_give_up_on_persistent_5xx():
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_retries=2)

    bad = _mock_response({"error": "boom"}, status=503)
    with patch.object(client._client, "post", return_value=bad), pytest.raises(EmbeddingError):
        client.embed_queries(["x"])


def test_retries_on_network_error():
    settings = _settings()
    client = EmbeddingClient(settings=settings, max_retries=3)

    side_effects = [
        httpx.ConnectError("connection refused"),
        _mock_response(_ok_payload(1, settings.dim)),
    ]
    with patch.object(client._client, "post", side_effect=side_effects):
        out = client.embed_queries(["x"])
    assert len(out) == 1


# ---------------------------------------------------------------- bad shapes


def test_unexpected_response_shape_raises_embedding_error():
    settings = _settings()
    client = EmbeddingClient(settings=settings)

    bad_shape = _mock_response({"unexpected": "field"})
    with patch.object(client._client, "post", return_value=bad_shape), pytest.raises(
        EmbeddingError, match="unexpected embed response"
    ):
        client.embed_queries(["x"])


# ------------------------------------------------------------- health_check


def test_health_check_ok_path():
    settings = _settings(dim=1536)
    client = EmbeddingClient(settings=settings)

    with patch.object(client._client, "post", return_value=_mock_response(_ok_payload(1, 1536))):
        info = client.health_check()
    assert info["status"] == "ok"
    assert info["dim_observed"] == 1536
    assert info["dim_match"] is True


def test_health_check_returns_down_on_error():
    settings = _settings()
    client = EmbeddingClient(settings=settings)

    with patch.object(client._client, "post", side_effect=httpx.ConnectError("boom")):
        info = client.health_check()
    assert info["status"] == "down"
    assert "ConnectError" in info["error"] or "boom" in info["error"]
