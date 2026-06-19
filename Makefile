.PHONY: help install qdrant-up qdrant-down qdrant-health embed-up embed-down rerank-up rerank-down up down marker ingest query eval test lint clean

PYTHON ?= python3
COMPOSE ?= docker compose

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
	$(PYTHON) -m pip install -e ".[dev,otel]"

qdrant-up:
	$(COMPOSE) --profile qdrant up -d qdrant
	@echo "Waiting for Qdrant to be healthy..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		if curl -sf -H "api-key: $${QDRANT_API_KEY:-dev-local-changeme}" http://localhost:6333/healthz >/dev/null; then \
			echo "Qdrant is up"; exit 0; \
		fi; sleep 1; done; \
		echo "Qdrant did not become healthy in time"; exit 1

qdrant-down:
	$(COMPOSE) stop qdrant

qdrant-health:
	@bash scripts/00_verify_qdrant.sh

embed-health:
	@bash scripts/00_verify_embed.sh

health: qdrant-health embed-health
	@echo "All services healthy."

embed-up:
	$(COMPOSE) --profile embed up -d tei-embed

embed-down:
	$(COMPOSE) stop tei-embed

rerank-up:
	$(COMPOSE) --profile rerank up -d tei-rerank

rerank-down:
	$(COMPOSE) stop tei-rerank

up: qdrant-up embed-up rerank-up
	@echo "All services started. Check 'docker compose ps'."

down:
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
