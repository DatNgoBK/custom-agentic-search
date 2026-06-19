"""Shared OpenViking bootstrap used by every CLI script.

Loads .env, materializes ov.conf with secrets expanded, sets the env var
that OpenViking uses to find the config, and yields an initialized client.
"""
from __future__ import annotations

import os
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openviking.sync_client import SyncOpenViking

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OV_CONF = REPO_ROOT / "ov.conf"


@contextmanager
def open_viking_session(
    *,
    ov_conf: Path | None = None,
    require_healthy: bool = True,
) -> Iterator[SyncOpenViking]:
    """Yield a ready-to-use SyncOpenViking client, then close it.

    Use this in every script so the bootstrap order (load env → materialize
    config → set env var → import openviking) stays consistent.
    """
    load_dotenv()

    # Local imports — must happen *after* load_dotenv() so settings see the env.
    from rag_qdrant.ingestion.ov_bootstrap import materialize_ov_conf
    from rag_qdrant.observability import configure_logging
    from rag_qdrant.settings import get_settings

    settings = get_settings()
    configure_logging(settings.obs.log_level)

    materialized = materialize_ov_conf(ov_conf or DEFAULT_OV_CONF)
    os.environ["OPENVIKING_CONFIG_FILE"] = str(materialized)

    from openviking.sync_client import SyncOpenViking  # noqa: PLC0415

    workspace = Path(settings.openviking.data_path).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    client = SyncOpenViking(path=str(workspace))
    try:
        client.initialize()
        if require_healthy and not client.is_healthy():
            raise RuntimeError("OpenViking client failed health check")
        yield client
    finally:
        # Best-effort: OpenViking 0.4.2 has known async-task quirks at shutdown.
        with suppress(Exception):
            client.close()
