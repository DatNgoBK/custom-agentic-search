"""Tests for ov.conf materialization with env expansion + optional sections."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag_qdrant.ingestion.ov_bootstrap import (
    _drop_comments,
    expand_env,
    materialize_ov_conf,
)


def test_expand_env_substitutes_vars(monkeypatch):
    monkeypatch.setenv("MY_VAR", "hello")
    assert expand_env("${MY_VAR} world") == "hello world"


def test_expand_env_recurses_into_dicts(monkeypatch):
    monkeypatch.setenv("X", "value")
    out = expand_env({"key": "${X}", "nested": {"deeper": "${X}!"}})
    assert out == {"key": "value", "nested": {"deeper": "value!"}}


def test_expand_env_recurses_into_lists(monkeypatch):
    monkeypatch.setenv("Y", "item")
    assert expand_env(["${Y}", "literal", "${Y}"]) == ["item", "literal", "item"]


def test_expand_env_passes_through_non_string():
    assert expand_env(42) == 42
    assert expand_env(True) is True
    assert expand_env(None) is None


def test_expand_env_raises_on_missing_required_var(monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    with pytest.raises(KeyError):
        expand_env("${DEFINITELY_NOT_SET_VAR}")


def test_drop_comments_removes_comment_keys():
    input_data = {
        "_comment": "should be dropped",
        "_comment_anything": "also dropped",
        "real_key": "kept",
        "nested": {
            "_comment": "nested dropped",
            "value": "kept",
        },
    }
    out = _drop_comments(input_data)
    assert "_comment" not in out
    assert "_comment_anything" not in out
    assert out["real_key"] == "kept"
    assert "_comment" not in out["nested"]
    assert out["nested"]["value"] == "kept"


# ------------------------------------------------------------ optional sections


def _make_minimal_template(tmp_path: Path, *, with_rerank: bool = True) -> Path:
    """Build a minimal ov.conf template for testing."""
    config = {
        "embedding": {"provider": "openai", "api_key": "${TEST_EMBED_KEY}"},
    }
    if with_rerank:
        config["rerank"] = {
            "provider": "cohere",
            "api_key": "${TEST_RERANK_KEY}",
            "model_name": "rerank-v3.5",
        }
    template = tmp_path / "ov.conf"
    template.write_text(json.dumps(config), encoding="utf-8")
    return template


def test_rerank_section_dropped_when_key_empty(tmp_path: Path, monkeypatch):
    template = _make_minimal_template(tmp_path)
    monkeypatch.setenv("TEST_EMBED_KEY", "embed-key")
    monkeypatch.delenv("TEST_RERANK_KEY", raising=False)  # not set → defaults to ""

    out = materialize_ov_conf(template, write_to=tmp_path / "out.json")
    data = json.loads(out.read_text())

    assert "rerank" not in data, "rerank section should be dropped when api_key is empty"
    assert data["embedding"]["api_key"] == "embed-key"


def test_rerank_section_kept_when_key_set(tmp_path: Path, monkeypatch):
    template = _make_minimal_template(tmp_path)
    monkeypatch.setenv("TEST_EMBED_KEY", "embed-key")
    monkeypatch.setenv("TEST_RERANK_KEY", "cohere-key-12345")

    out = materialize_ov_conf(template, write_to=tmp_path / "out.json")
    data = json.loads(out.read_text())

    assert "rerank" in data
    assert data["rerank"]["api_key"] == "cohere-key-12345"
    assert data["rerank"]["model_name"] == "rerank-v3.5"


def test_required_var_still_raises(tmp_path: Path, monkeypatch):
    """Required vars (in non-optional sections) must still fail loudly."""
    template = tmp_path / "ov.conf"
    template.write_text(
        json.dumps({"storage": {"url": "${REQUIRED_BUT_UNSET}"}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("REQUIRED_BUT_UNSET", raising=False)
    # Need to also clean any leftover from earlier tests
    os.environ.pop("REQUIRED_BUT_UNSET", None)

    with pytest.raises(KeyError):
        materialize_ov_conf(template, write_to=tmp_path / "out.json")


def test_comments_stripped_from_materialized(tmp_path: Path, monkeypatch):
    template = tmp_path / "ov.conf"
    template.write_text(
        json.dumps({
            "_comment": "doc",
            "embedding": {"_comment": "doc2", "api_key": "${E}"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("E", "v")
    out = materialize_ov_conf(template, write_to=tmp_path / "out.json")
    data = json.loads(out.read_text())
    assert "_comment" not in data
    assert "_comment" not in data["embedding"]
