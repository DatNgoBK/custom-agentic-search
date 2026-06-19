"""Bootstrap helpers: materialize ov.conf with secrets resolved from env.

We don't commit secrets into ``ov.conf``. The template uses ``${VAR}``
placeholders which we expand at load time using ``string.Template`` —
the same syntax shells use, so the file doubles as documentation.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from string import Template
from typing import Any

from rag_qdrant.observability import get_logger

log = get_logger("rag_qdrant.bootstrap")

# Optional sections: if their api_key resolves to empty, the entire section
# is dropped from the final config so OpenViking treats them as "not configured"
# instead of crashing on missing-env errors. Required services (Qdrant URL,
# embedding API key) still raise as before.
_OPTIONAL_SECTIONS = ("rerank",)


def expand_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` strings with their environment values.

    Empty env vars (``""``) substitute through normally; missing env vars
    raise ``KeyError`` so misconfiguration fails at load time. Strings with
    no placeholders pass through unchanged.
    """
    if isinstance(value, str):
        # safe_substitute leaves missing ${VAR} as-is; we use strict substitute
        # to fail fast — but we set unset vars to "" beforehand for optional ones.
        return Template(value).substitute(os.environ)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def materialize_ov_conf(template_path: Path, *, write_to: Path | None = None) -> Path:
    """Load ``ov.conf`` template, expand env vars, write a runtime copy.

    Optional sections (``rerank``) with an empty api_key are stripped so
    OpenViking sees them as not configured. Documentation-only keys (any
    starting with ``_comment``) are also stripped. Secrets land in a
    tempfile, not the repo, so they never reach git.
    """
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    cleaned = _drop_comments(raw)

    # Default optional env vars to empty so substitution doesn't KeyError
    # when a user hasn't filled them in.
    _ensure_optional_env_defaults(cleaned)

    expanded = expand_env(cleaned)
    expanded = _drop_unconfigured_optional_sections(expanded)

    if write_to is None:
        fd, tmp_path = tempfile.mkstemp(prefix="ov_conf_", suffix=".json")
        os.close(fd)
        write_to = Path(tmp_path)

    write_to.write_text(json.dumps(expanded, indent=2), encoding="utf-8")
    log.info("ov_conf.materialized", path=str(write_to), template=str(template_path))
    return write_to


def _ensure_optional_env_defaults(config: dict[str, Any]) -> None:
    """Set missing env vars referenced by optional sections to empty string.

    Prevents ``string.Template.substitute`` from raising on unset COHERE_API_KEY
    (etc.) when the user just wants to skip that feature.
    """
    import re
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")
    for section_name in _OPTIONAL_SECTIONS:
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for value in section.values():
            if isinstance(value, str):
                for var in pattern.findall(value):
                    os.environ.setdefault(var, "")


def _drop_unconfigured_optional_sections(config: dict[str, Any]) -> dict[str, Any]:
    """Remove optional sections whose api_key resolved to an empty string."""
    result = dict(config)
    for section_name in _OPTIONAL_SECTIONS:
        section = result.get(section_name)
        if isinstance(section, dict) and not section.get("api_key"):
            result.pop(section_name, None)
            log.info("ov_conf.optional_section_skipped", section=section_name)
    return result


def _drop_comments(value: Any) -> Any:
    """Remove keys whose name starts with ``_comment`` from any nested dict."""
    if isinstance(value, dict):
        return {
            k: _drop_comments(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.startswith("_comment"))
        }
    if isinstance(value, list):
        return [_drop_comments(v) for v in value]
    return value
