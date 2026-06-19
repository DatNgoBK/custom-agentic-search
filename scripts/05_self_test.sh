#!/usr/bin/env bash
# Self-test the full pipeline as a reviewer would experience it.
#
# Runs in 7 phases, each fails the script (set -e) on the first error.
# Designed to mirror the README's Quickstart path so any breakage we
# catch here is something a reviewer would also hit.
#
# Usage:
#   bash scripts/05_self_test.sh           # full run
#   FAST=1 bash scripts/05_self_test.sh    # skip slow phases (ingest)
set -euo pipefail

cd "$(dirname "$0")/.."

bold()  { printf "\033[1m\n══ %s ══\033[0m\n" "$*"; }
ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$*"; }
fail()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ---------- Phase 1: Static checks ----------
bold "Phase 1 — Static checks (no services needed)"

[[ -f README.md ]]              || fail "README.md missing"
[[ -f Makefile ]]               || fail "Makefile missing"
[[ -f docker-compose.yml ]]     || fail ".env.example missing"
[[ -f .env.example ]]           || fail ".env.example missing"
[[ -f ov.conf ]]                || fail "ov.conf missing"
[[ -f pyproject.toml ]]         || fail "pyproject.toml missing"
ok "Required top-level files present"

[[ -d rag_qdrant/adapters ]]    || fail "rag_qdrant/adapters missing"
[[ -d rag_qdrant/cli ]]         || fail "rag_qdrant/cli missing"
[[ -d rag_qdrant/preprocessing ]] || fail "rag_qdrant/preprocessing missing"
[[ -d tests/adapters ]]         || fail "tests/adapters missing"
ok "Package structure intact"

# Marker pre-processed output should be in the repo so reviewers can skip Marker
[[ -f output/source/source.md ]]         || fail "output/source/source.md missing (reviewers can't skip Marker)"
[[ -d output/source/chunks ]]            || fail "output/source/chunks/ missing"
chunk_count=$(find output/source/chunks -name '*.md' | wc -l | tr -d ' ')
[[ $chunk_count -gt 100 ]]               || fail "Only $chunk_count chunks; expected 300+"
ok "Marker output committed ($chunk_count chunks)"

# Private/personal docs MUST NOT exist as committable files
if [[ -d private && ! -f .gitignore ]]; then
    fail "private/ exists but .gitignore missing"
fi
if [[ -f .gitignore ]] && ! grep -q '^private/' .gitignore; then
    fail ".gitignore missing 'private/' rule"
fi
ok "Private docs gitignored"

# .env must NOT contain placeholder when ready for use
if grep -q "PUT-YOUR-OPENROUTER-KEY" .env 2>/dev/null; then
    warn ".env still has placeholder EMBED_API_KEY (OK if reviewer just copied .env.example)"
fi

# ---------- Phase 2: Python install ----------
bold "Phase 2 — Python environment"

if [[ ! -d .venv ]]; then
    warn ".venv missing — creating one"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv activated"

if ! python -c "import rag_qdrant" 2>/dev/null; then
    warn "rag_qdrant not installed — running pip install -e .[dev]"
    pip install -e ".[dev]" >/dev/null 2>&1 || fail "pip install failed"
fi
python -c "import rag_qdrant; import openviking; import qdrant_client" 2>/dev/null \
    || fail "Required packages missing (rag_qdrant / openviking / qdrant-client)"
ok "All required imports work"

# ---------- Phase 3: Lint + unit tests ----------
bold "Phase 3 — Lint + unit tests (no services needed)"

ruff check rag_qdrant tests scripts >/dev/null 2>&1 \
    && ok "ruff: clean" \
    || fail "ruff check failed — run 'make lint' for details"

pytest tests/ -q >/tmp/pytest.out 2>&1 \
    && ok "pytest: $(grep -oE '[0-9]+ passed' /tmp/pytest.out | tail -1)" \
    || { tail -20 /tmp/pytest.out >&2; fail "pytest failed"; }

# ---------- Phase 4: Qdrant smoke ----------
bold "Phase 4 — Qdrant service"

if ! docker info >/dev/null 2>&1; then
    fail "Docker daemon not running. Start Docker Desktop, then re-run."
fi
ok "Docker is running"

if ! docker ps --filter name=rag-qdrant --format '{{.Status}}' | grep -q "healthy\|Up"; then
    warn "Qdrant not running — starting via 'make qdrant-up'"
    make qdrant-up >/dev/null 2>&1 || fail "make qdrant-up failed"
fi

bash scripts/00_verify_qdrant.sh >/dev/null 2>&1 \
    && ok "Qdrant: healthy + auth enforced" \
    || fail "Qdrant smoke test failed — check 'make qdrant-health'"

# ---------- Phase 5: Embedding endpoint ----------
bold "Phase 5 — Embedding endpoint (OpenRouter)"

# Source .env if present so EMBED_API_KEY is loaded
[[ -f .env ]] && set -a && source .env && set +a

if [[ -z "${EMBED_API_KEY:-}" || "${EMBED_API_KEY}" == *"PUT-YOUR-OPENROUTER-KEY"* ]]; then
    fail "EMBED_API_KEY not set in .env (still placeholder?). Edit .env and paste your OpenRouter key."
fi

bash scripts/00_verify_embed.sh >/dev/null 2>&1 \
    && ok "Embed endpoint: 200, dim 1536, multilingual works" \
    || fail "Embed smoke test failed — check 'make embed-health'"

# ---------- Phase 6: Bootstrap config ----------
bold "Phase 6 — ov.conf materialization"

python -c "
from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from rag_qdrant.ingestion.ov_bootstrap import materialize_ov_conf
import json
out = materialize_ov_conf(Path('ov.conf'))
data = json.loads(out.read_text())
assert 'storage' in data, 'storage section missing'
assert 'embedding' in data, 'embedding section missing'
assert 'rerank' in data, 'rerank should be present (EMBED_API_KEY set)'
assert data['embedding']['dense']['api_key'].startswith('sk-or-v1-'), 'embed api_key not expanded'
print('OK')
" >/dev/null 2>&1 && ok "ov.conf materializes with rerank enabled" \
    || fail "ov.conf materialization broken"

# ---------- Phase 7: Query end-to-end ----------
bold "Phase 7 — Query end-to-end"

if [[ ! -f .ingestion_state.json ]]; then
    if [[ "${FAST:-0}" == "1" ]]; then
        warn "No .ingestion_state.json and FAST=1 set — skipping ingest"
        warn "Skipping query phase too (no data to query)"
    else
        warn "No .ingestion_state.json — running 'make ingest' first (~3-4 min)"
        make ingest >/tmp/ingest.out 2>&1 \
            || { tail -20 /tmp/ingest.out >&2; fail "make ingest failed"; }
        ok "Ingest completed"
    fi
fi

if [[ -f .ingestion_state.json ]]; then
    # Use venv python explicitly (make invokes python3 from PATH which may
    # not be the venv even after `source` due to subshell isolation).
    PY="${VIRTUAL_ENV:-$PWD/.venv}/bin/python"
    "$PY" scripts/03_test_query.py --query "Lợi nhuận MSB 2024?" --limit 2 >/tmp/query.out 2>&1 \
        && grep -q "queries returned at least one hit" /tmp/query.out \
        && ok "Query end-to-end: 1/1 hit returned" \
        || { tail -20 /tmp/query.out >&2; fail "Query failed"; }
fi

bold "ALL CHECKS PASSED ✅"
echo "Repo is ready for reviewer hand-off."
