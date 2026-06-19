.PHONY: help install qdrant-up qdrant-down qdrant-health embed-up embed-down rerank-up rerank-down up down marker ingest query eval test lint clean self-test self-test-fast _check_compose

# After `make install`, .venv exists and we use its Python directly so
# users don't need to remember to `source .venv/bin/activate`. Override
# with `make ingest PYTHON=python3` if you've installed globally.
PYTHON  ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Detect Docker Compose flavor: v2 plugin (`docker compose`) is preferred,
# but fall back to v1 standalone (`docker-compose`) for users who haven't
# updated. If neither works, the qdrant-up target prints a clear hint.
COMPOSE ?= $(shell \
	if docker compose version >/dev/null 2>&1; then \
		echo "docker compose"; \
	elif command -v docker-compose >/dev/null 2>&1; then \
		echo "docker-compose"; \
	else \
		echo "missing-compose"; \
	fi)

# For `make install` we need a Python >= 3.10 binary. macOS often ships
# with 3.9 as `python3` so we try common 3.10+ binaries first and fall
# back to PYTHON if they aren't present. Override: `make install PYTHON_BOOTSTRAP=/path/to/python3.11`
PYTHON_BOOTSTRAP ?= $(shell command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 || echo missing-python)

help:
	@echo "Targets:"
	@echo "  install        — install package + dev deps"
	@echo "  qdrant-up      — start standalone Qdrant (Docker)"
	@echo "  qdrant-down    — stop Qdrant"
	@echo "  qdrant-health  — verify Qdrant is up + auth working"
	@echo "  embed-health   — verify embedding endpoint reachable"
	@echo "  health         — qdrant-health + embed-health"
	@echo ""
	@echo "  marker         — run Marker on input PDF (native, MPS, ~30 min)"
	@echo "  ingest         — preprocess + ingest markdown into Qdrant"
	@echo "  quality-check  — validate preprocessed chunks"
	@echo "  query          — run 8 demo queries (Vietnamese)"
	@echo "  query-i        — interactive REPL: type questions, get answers"
	@echo "  ask Q=\"...\"    — one-shot custom query"
	@echo ""
	@echo "  test           — pytest (47 unit tests)"
	@echo "  lint           — ruff + mypy"
	@echo "  clean          — remove caches and runtime data"
	@echo "  clean-data     — remove ingested data (forces re-ingest)"
	@echo "  demo           — full e2e: up + ingest + quality-check + query"

install:
	@# 1. Verify a usable Python interpreter is available
	@if [ "$(PYTHON_BOOTSTRAP)" = "missing-python" ]; then \
		echo "::error:: No python3 binary found on PATH."; \
		echo "  This project needs Python 3.10 or newer."; \
		echo "  macOS:  brew install python@3.12"; \
		echo "  Ubuntu: sudo apt install python3.12 python3.12-venv"; \
		echo "  Or download from https://www.python.org/downloads/"; \
		exit 1; \
	fi
	@# 2. Verify the version is >= 3.10
	@$(PYTHON_BOOTSTRAP) -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || { \
		ver=$$($(PYTHON_BOOTSTRAP) --version 2>&1); \
		echo "::error:: $(PYTHON_BOOTSTRAP) reports $$ver, but project requires Python >= 3.10."; \
		echo ""; \
		echo "  Dependencies (openviking, qdrant-client, marker-pdf, ...) all"; \
		echo "  require 3.10+, so we cannot relax this."; \
		echo ""; \
		echo "  Easiest fix on macOS:"; \
		echo "      brew install python@3.12"; \
		echo "      make install PYTHON_BOOTSTRAP=$$(brew --prefix python@3.12)/bin/python3.12"; \
		echo ""; \
		echo "  Already have a newer Python at a custom path?"; \
		echo "      make install PYTHON_BOOTSTRAP=/path/to/python3.12"; \
		exit 1; \
	}
	@# 3. Create venv + install
	@if [ ! -d .venv ]; then \
		echo "→ Creating .venv with $$($(PYTHON_BOOTSTRAP) --version 2>&1)..."; \
		$(PYTHON_BOOTSTRAP) -m venv .venv; \
	fi
	@.venv/bin/python -m pip install --upgrade pip --quiet
	@.venv/bin/python -m pip install -e ".[dev,otel]"
	@echo
	@echo "✓ Installed into .venv. All make targets auto-use it."

qdrant-up: _check_compose
	$(COMPOSE) up -d qdrant
	@echo "Waiting for Qdrant to be healthy..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		if curl -sf -H "api-key: $${QDRANT_API_KEY:-dev-local-changeme}" http://localhost:6333/healthz >/dev/null; then \
			echo "Qdrant is up"; exit 0; \
		fi; sleep 1; done; \
		echo "Qdrant did not become healthy in time"; exit 1

qdrant-down: _check_compose
	$(COMPOSE) stop qdrant

qdrant-health:
	@bash scripts/00_verify_qdrant.sh

embed-health:
	@bash scripts/00_verify_embed.sh

health: qdrant-health embed-health
	@echo "All services healthy."

# Optional local-embed/rerank profiles (off by default — OpenRouter handles both).
embed-up: _check_compose
	$(COMPOSE) --profile local-embed up -d tei-embed

embed-down: _check_compose
	$(COMPOSE) stop tei-embed

rerank-up: _check_compose
	$(COMPOSE) --profile rerank up -d tei-rerank

rerank-down: _check_compose
	$(COMPOSE) stop tei-rerank

up: qdrant-up
	@echo "Services started. (Local TEI embed/rerank are optional; see 'make help')."

# Internal: bail with a clear message if Docker Compose isn't installed.
_check_compose:
	@if [ "$(COMPOSE)" = "missing-compose" ]; then \
		echo "::error:: Docker Compose not found."; \
		echo "  Install Docker Desktop (includes Compose v2):"; \
		echo "      https://www.docker.com/products/docker-desktop/"; \
		echo "  Or, on Linux without Desktop, install the plugin:"; \
		echo "      sudo apt install docker-compose-plugin"; \
		echo "  Or, legacy v1 standalone:"; \
		echo "      pip install docker-compose"; \
		exit 1; \
	fi

down: _check_compose
	$(COMPOSE) down

marker:
	bash scripts/01_run_marker.sh

ingest:
	$(PYTHON) scripts/02_ingest.py

quality-check:
	$(PYTHON) scripts/04_quality_check.py

query:
	$(PYTHON) scripts/03_test_query.py

query-i:
	$(PYTHON) scripts/03_test_query.py -i

ask:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make ask Q=\"câu hỏi của bạn\""; \
		exit 1; \
	fi
	@$(PYTHON) scripts/03_test_query.py --query "$(Q)"

eval:
	$(PYTHON) tests/eval/run_eval.py

test:
	pytest -v tests/

lint:
	ruff check rag_qdrant tests scripts
	mypy rag_qdrant

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	rm -rf data/ qdrant_storage/ output/chunks/ output/source/*_cleaned.md
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-data:
	rm -rf data/ qdrant_storage/

demo: up ingest quality-check query
	@echo "Demo complete."

self-test:
	@bash scripts/05_self_test.sh

self-test-fast:
	@FAST=1 bash scripts/05_self_test.sh
