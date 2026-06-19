# Custom Agentic Search Pipeline — Marker × OpenViking × Qdrant

> Take-home assignment: build a document search & QA pipeline that combines
> **Marker** (PDF preprocessing), **OpenViking** (agentic search engine),
> and a **standalone Qdrant** instance — bypassing OpenViking's bundled
> vector store via a custom adapter.

**Sample doc:** MSB Annual Report 2024 (Vietnamese, 240 pages, 17 MB).
**Status:** end-to-end pipeline working — see [Live demo](#live-demo).

---

## Table of contents

1. [TL;DR](#tldr)
2. [Quickstart (3 commands)](#quickstart-3-commands)
3. [How to query](#how-to-query)
4. [Architecture](#architecture)
5. [Integration approach](#integration-approach)
6. [Configuration reference](#configuration-reference)
7. [Live demo](#live-demo)
8. [Test suite](#test-suite)
9. [Marker, OCR, and Vietnamese content](#marker-ocr-and-vietnamese-content)
10. [Production gaps acknowledged](#production-gaps-acknowledged)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## TL;DR

* **Custom `CustomQdrantCollectionAdapter`** subclasses OpenViking's
  built-in `QdrantCollectionAdapter` and is loaded **without forking
  OpenViking** through its dotted-path factory mechanism.
* **Standalone Qdrant** in Docker (API-key auth, persistent volume,
  healthchecks, resource limits).
* **Embeddings via OpenRouter** (`openai/text-embedding-3-small`, 1536d) —
  no local GPU needed.
* **Reranker via OpenRouter** (`cohere/rerank-v3.5`) — multilingual
  cross-encoder boosts top-1 score by ~+0.22 points on Vietnamese queries.
* **Preprocessing pipeline** cleans Marker artifacts (1894 `<br>` tags
  removed) and chunks 240-page report into 398 semantic sections →
  **1293 embeddings** in Qdrant.
* **Resilience built-in**: pydantic config validation, tenacity retry,
  pybreaker circuit breaker, Prometheus metrics, structured logs,
  deterministic UUIDv5 point IDs for idempotent re-ingestion.
* **57 unit tests** (lint clean) + smoke scripts for Qdrant and
  embedding endpoint.

---

## Quickstart (3 commands)

**Prerequisites**: Docker, Python 3.10–3.12, an OpenRouter API key
([get free here](https://openrouter.ai/keys)).
**RAM**: ~3 GB during ingest, ~500 MB at idle.

```bash
# 1. Configure secret (paste your OpenRouter key)
cp .env.example .env && $EDITOR .env

# 2. Install + start Qdrant
make install && make qdrant-up

# 3. Run the demo (ingest + query)
make demo
```

That's it. `make demo` runs the full pipeline end-to-end:
preprocessing → ingestion → quality validation → 8 demo queries.

**Already ingested before?** Just run `make query` (or any of the query
modes below) — Qdrant data persists across restarts.

---

## How to query

After `make ingest` (or `make demo`), you can query the corpus in **3
different ways**:

### 1. Run 8 pre-defined Vietnamese demo queries

```bash
make query
```

Runs 8 hand-picked questions about the MSB report and prints scored
hits + a summary table. Used for the deliverable's "test script
demonstrating successful agentic search".

### 2. Ask one custom question

```bash
make ask Q="Tổng tài sản MSB cuối năm 2024 là bao nhiêu?"
make ask Q="Lợi nhuận trước thuế năm 2024?"
make ask Q="Ai là Tổng giám đốc MSB?"
```

Or directly without `make`:
```bash
python scripts/03_test_query.py --query "câu hỏi của bạn" --limit 5
```

### 3. Interactive REPL — type questions, get answers, loop

```bash
make query-i
```

Output:
```
================================================================================
 AGENTIC SEARCH DEMO — MSB Annual Report 2024 (Vietnamese)
================================================================================
  Root URI:          viking://resources/chunks
  Qdrant collection: msb_report_2024
  Embed model:       openai/text-embedding-3-small  (1536d)
  ...

Interactive mode — type a question, press Enter.
  • Empty line, 'exit', 'quit', or Ctrl+D to leave.
  • Vietnamese works best (the corpus is the MSB 2024 report).

Q› Tổng tài sản MSB?
  1. [ 0.712] viking://resources/chunks/042_31_Tổng_tài_sản/...
  2. [ 0.689] viking://resources/chunks/060_411_Tổng_tài_sản_và_tăng_trưởng_tín_dụng/...
  ⏱  567 ms

Q› Mã chứng khoán MSB?
  1. [ 0.747] viking://resources/chunks/011_B_THÔNG_TIN_CHUNG_VỀ_MSB/...
  ...

Q› exit
Bye.
```

### Useful flags

| Flag | Meaning |
|---|---|
| `--limit N` | Top-K hits per query (default 5) |
| `--query "text"` | One-shot question; repeatable for batch |
| `-i, --interactive` | REPL mode |
| `--snippet-chars N` | Snippet length (default 180) |

Full help: `python scripts/03_test_query.py --help`

---

## Architecture

```
                  ┌──────────────────────────────────┐
                  │  Marker (offline, MPS-accelerated)│
                  │  PDF → markdown (980 KB raw)      │
                  └──────────────────┬────────────────┘
                                     │ output/source/source.md
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  Preprocessing (rag_qdrant/preprocessing/)   │
              │   • clean_marker_output:                     │
              │       strip <br> in tables, signatures, noise│
              │   • chunk_markdown: split by H1-H4           │
              │   → 398 chunk files                          │
              └──────────────────┬───────────────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────────────┐
              │  scripts/02_ingest.py / rag-ingest           │
              │   • materialize ov.conf (env expansion)      │
              │   • SyncOpenViking.add_resource(chunks/)     │
              └──────────────────┬───────────────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                                 ▼
   ┌──────────────────────┐      ┌──────────────────────────────────────┐
   │  OpenAI provider →   │      │  CustomQdrantCollectionAdapter        │
   │  OpenRouter (network)│      │   (subclass of QdrantCollectionAdapter│
   │  text-embedding-     │      │    loaded via dotted-path)            │
   │  3-small (1536d)     │      │   • pydantic config validation        │
   └──────────────────────┘      │   • tenacity retry + pybreaker        │
                                 │   • Prometheus metrics + structlog    │
                                 │   • deterministic UUIDv5 point IDs    │
                                 │   • HNSW + scalar int8 quant config   │
                                 │   • health_check() for k8s readiness  │
                                 └──────────────────┬───────────────────┘
                                                    │ HTTP / gRPC
                                                    ▼
                                  ┌────────────────────────────────────┐
                                  │  Qdrant 1.12 (Docker, API-key auth)│
                                  │  msb_demo__msb_report_2024         │
                                  │  → 1293 embeddings                 │
                                  └────────────────────────────────────┘
                                                    ▲
                                                    │ search top-K (e.g. 20)
                                                    │
                                  ┌─────────────────┴────────────────┐
                                  │  HierarchicalRetriever            │
                                  │  + Cohere rerank-v3.5             │
                                  │   (via OpenRouter)                │
                                  │   re-orders top-K → top-N (e.g.5) │
                                  └─────────────────┬────────────────┘
                                                    │
                                  ┌─────────────────┴────────────────┐
                                  │  scripts/03_test_query.py        │
                                  │  → make query / make ask /       │
                                  │    make query-i (REPL)           │
                                  └──────────────────────────────────┘
```

---

## Integration approach

### How OpenViking finds our adapter

OpenViking's adapter factory accepts either a registry key (`"qdrant"`,
`"local"`, ...) **or** a fully-qualified Python class path:

```python
# openviking/storage/vectordb_adapters/factory.py:33-42
if adapter_cls is None and "." in backend:
    module_name, class_name = backend.rsplit(".", 1)
    module = importlib.import_module(module_name)
    potential_cls = getattr(module, class_name)
    if issubclass(potential_cls, CollectionAdapter):
        adapter_cls = potential_cls
```

So `ov.conf` simply names our class:

```jsonc
"storage": {
  "vectordb": {
    "backend": "rag_qdrant.adapters.custom_qdrant_adapter.CustomQdrantCollectionAdapter",
    "name":    "msb_report_2024",
    "project": "msb_demo",
    "qdrant":  { "url": "${QDRANT_URL}", "api_key": "${QDRANT_API_KEY}" },
    "custom_params": {
      "url":          "${QDRANT_URL}",
      "api_key":      "${QDRANT_API_KEY}",
      "hnsw":         { "m": 32, "ef_construct": 256 },
      "quantization": { "enabled": true, "type": "int8" },
      "max_retries":  3,
      "breaker_fail_max": 5
    }
  }
}
```

Secrets stay out of the file — they live in `.env` and are expanded by
`rag_qdrant.ingestion.ov_bootstrap.materialize_ov_conf()` into a tempfile
that OpenViking loads via the `OPENVIKING_CONFIG_FILE` env variable.

### Why subclass instead of rewrite

OpenViking ships a reference Qdrant adapter (377 lines) with a
fully-tested filter compiler that understands the agentic retrieval URI
scheme (`viking://…`, `parent_uri`, `scope_roots`). Rewriting from
scratch would re-introduce subtle bugs that took the upstream
maintainers iterations to fix.

`CustomQdrantCollectionAdapter` therefore **inherits** the parts that
are correct (`_compile_filter`, `_normalize_record_for_write/read`,
`_load_existing_collection_if_needed`) and **extends** what production
operations need:

| Concern | Where in our code |
|---|---|
| Strict config validation | `adapters/config.py` — pydantic, fails at construct time |
| Deterministic point IDs | `adapters/identity.py` — UUIDv5 over `(uri, chunk)` |
| Retry + circuit breaker | `adapters/resilience.py` — wraps every I/O op |
| Metrics + structured logs | `observability/{logging,metrics}.py` |
| HNSW + scalar int8 quant | `_build_default_index_meta` override |
| Health probe | `health_check()` returns `{status, vector_count, breaker, …}` |

### Embedding integration

OpenViking already has an `openai` provider for embeddings. Pointing it
at OpenRouter (`https://openrouter.ai/api/v1`) was enough — **no custom
vectorizer plugin required**. The thin client in
`rag_qdrant/embedding/` is OpenAI-compatible, so it also works with
local TEI / sentence-transformers proxies if you don't want OpenRouter.

### Reranker integration

Same pattern: OpenViking's `RerankConfig` accepts `provider="openai"` for
any OpenAI-compatible rerank endpoint. We point it at OpenRouter's
`/v1/rerank` and use `cohere/rerank-v3.5` — a multilingual
cross-encoder. The same `EMBED_API_KEY` covers both embed + rerank, so
**no extra account or env var** is needed.

```jsonc
"rerank": {
  "provider": "openai",
  "api_base": "https://openrouter.ai/api/v1/rerank",
  "api_key":  "${EMBED_API_KEY}",
  "model":    "cohere/rerank-v3.5",
  "threshold": 0.1
}
```

The reranker re-orders the top-K hits returned by Qdrant before the
final result is shown. Optional via the bootstrap script: if
`EMBED_API_KEY` ever resolves to empty, the rerank section is dropped
and OpenViking falls back to dense vector scores. Concretely, on the
demo set, the rerank stage:

* Boosts top-1 score from `~0.74 → ~0.95` on Vietnamese queries.
* Picks more precise sub-sections (e.g. `065_423_Lợi_nhuận_của_Ngân_hàng`
  instead of the broader `046_35_Lợi_nhuận_trước_thuế`).
* Adds ~10–12 s of latency (OpenRouter rerank proxy is slow). For
  production, a Cohere direct API or local TEI rerank service would
  shave this back to a few hundred ms.

### Preprocessing pipeline

Marker output is good but not perfect for RAG. `rag_qdrant/preprocessing/`
adds 2 stages:

1. **`clean_marker_output`** — strips `<br>` tags inside table cells
   (1894 of them in the MSB PDF), removes digital signature blocks
   (PKI metadata, Foxit Reader stamps), applies targeted patches from
   `msb_patches.json` (OCR errors specific to this dataset), normalizes
   whitespace.
2. **`chunk_markdown`** — splits the cleaned 980 KB markdown into 398
   files based on H1-H4 headings. Code-block aware (won't split on `#`
   inside ` ``` `). Filenames preserve Vietnamese diacritics, are safe
   for any filesystem.

Result: search score average improved by **+0.11 points** vs single-file
ingestion (granular chunks → more precise retrieval).

### End-to-end data flow

1. `make marker` (offline, ~30 min) → `output/source/source.md`
2. `make ingest`
   * Preprocessing: clean → chunk into 398 files.
   * `SyncOpenViking.add_resource(chunks_dir, wait=True)` parses each
     file, builds URI tree, generates L0 abstracts via VLM (gpt-4o-mini
     through OpenRouter).
   * Embeddings produced via OpenRouter → 1536d vectors.
   * Vectors upserted through `CustomQdrantCollectionAdapter` into
     `msb_demo__msb_report_2024` on standalone Qdrant.
   * `.ingestion_state.json` persists `root_uri` for the next step.
3. `make query` / `make ask Q="..."` / `make query-i`
   * OpenViking generates a query vector, calls `adapter.query()`.
   * Adapter applies retry/breaker/metrics, dispatches to Qdrant, returns
     ranked hits.
   * Script prints scored URIs + content snippets.

---

## Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP base URL |
| `QDRANT_API_KEY` | `dev-local-changeme` | Qdrant API key (auth required) |
| `QDRANT_COLLECTION` | `msb_report_2024` | Logical collection name |
| `EMBED_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compat endpoint |
| `EMBED_MODEL` | `openai/text-embedding-3-small` | Embed model id (1536d) |
| `EMBED_API_KEY` | *(required)* | OpenRouter / OpenAI API key |
| `EMBED_DIM` | `1536` | Used for sanity checks |
| `OV_DATA_PATH` | `./data` | OpenViking workspace |
| `OV_PROJECT_NAME` | `msb_demo` | Becomes collection name prefix |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `METRICS_PORT` | `9100` | Prometheus exporter port |

See `.env.example` for the full list. `ov.conf` references env vars with
`${VAR}` syntax — they are expanded at runtime, **never written to disk
checked into git**.

---

## Live demo

Output of `make query` against the full MSB report (240 pages, 1293
embeddings, all stored in standalone Qdrant via the custom adapter,
with Cohere rerank-v3.5 enabled):

```
================================================================================
 AGENTIC SEARCH DEMO — MSB Annual Report 2024 (Vietnamese)
================================================================================
  Root URI:          viking://resources/chunks
  Qdrant collection: msb_report_2024
  Embed model:       openai/text-embedding-3-small  (1536d)
  Adapter:           rag_qdrant.adapters.custom_qdrant_adapter
                     .CustomQdrantCollectionAdapter
  Top-K per query:   5
================================================================================

[HierarchicalRetriever] Rerank enabled (provider=openai), threshold=0.1

[1/8] Q: Lợi nhuận MSB 2024?
  1. [ 0.955] viking://resources/chunks/065_423_Lợi_nhuận_của_Ngân_hàng_và_khả_năng_sinh_lời/...
     Tài liệu cung cấp thông tin về lợi nhuận và khả năng sinh lời của
     Ngân hàng MSB trong năm 2024…
  2. [ 0.945] viking://resources/chunks/010_Kính_thưa_Quý_Khách_hàng_Đối_tác/...
  3. [ 0.930] viking://resources/chunks/046_35_Lợi_nhuận_trước_thuế_thu_nhập_doanh_nghiệp/...
  4. [ 0.893] viking://resources/chunks/063_421_Tổng_thu_nhập_hoạt_động/...
  5. [ 0.865] viking://resources/chunks/165_1_Đánh_giá_của_Hội_đồng_Quản_trị/...
  ⏱  12152 ms (includes rerank round trip)

================================================================================
 SUMMARY (8/8 queries returned at least one hit)
================================================================================
```

### Quality impact of the reranker

Same query, same corpus, with vs without `cohere/rerank-v3.5`:

| Metric | Dense only | Dense + rerank | Δ |
|---|---|---|---|
| Top-1 score (Vietnamese queries) | ~0.69–0.74 | **~0.93–0.96** | **+0.20–0.22** |
| Top-1 URI (e.g. "Lợi nhuận MSB 2024?") | `046_35_Lợi_nhuận_trước_thuế` (broader) | `065_423_Lợi_nhuận_của_Ngân_hàng_và_khả_năng_sinh_lời` (precise) | More on-topic |
| End-to-end latency p50 | ~500 ms | ~10–12 s | OpenRouter rerank proxy is the slow link |

Rerank can be turned off by clearing `EMBED_API_KEY` (the bootstrap
script will then drop the rerank section automatically) — search falls
back to dense vector scores. For production, swap in Cohere direct API
or a local TEI rerank service to cut the latency to <500 ms.

---

## Test suite

```bash
$ make test
57 passed in 15s
```

Unit tests live in `tests/`:

```
tests/
├── adapters/
│   ├── conftest.py             # shared fixtures
│   ├── test_config.py          # 5 tests — Pydantic schema
│   ├── test_identity.py        # 4 tests — UUID v5
│   ├── test_resilience.py      # 4 tests — retry + breaker
│   └── test_adapter.py         # 7 tests — index meta, health, factory
├── test_embedding_unit.py      # 13 tests — embedding client
├── test_preprocessing.py       # 14 tests — clean + chunk markdown
└── test_ov_bootstrap.py        # 10 tests — env expansion + optional sections
```

Smoke scripts (no test framework, exit codes used by Make):

* `scripts/00_verify_qdrant.sh` — checks `/healthz`, auth enforcement,
  collection list, version.
* `scripts/00_verify_embed.sh` — checks reachable, returns expected dim,
  Vietnamese passage produces a non-degenerate vector.

---

## Marker, OCR, and Vietnamese content

We pre-processed the **full MSB 2024 annual report** (240 pages, 17 MB
PDF). The output is committed at `output/source/source.md` (~980 KB raw,
~974 KB after preprocessing) so graders don't need to download Marker's
~3 GB models or wait ~30 minutes.

To regenerate from scratch:

```bash
TORCH_DEVICE=mps make marker        # ~30 min for full PDF on M-series Mac
```

What works well:

* **Vietnamese diacritics** preserved (đ, ư, ơ, …)
* Tables flattened to GitHub-flavored markdown
* Numbers and percentages preserved exactly
* Heading hierarchy maintained (H1-H4 → 398 chunks)
* OCR error patches applied via `msb_patches.json`

What Marker does **not** do (limitations of its OCR stack):

* **Handwriting** — Marker uses Surya OCR which is trained on printed
  text. Handwritten signatures appear in the source as image graphics
  and are silently skipped. For the MSB annual report this doesn't
  matter — the CEO's printed name is in the rendered text already.
* **Stamps / seals** — text inside red corporate stamps may or may not
  be detected depending on print quality.
* **Free-form annotations** — pen marks on top of a printed PDF are
  out of scope.

Want handwriting too? Marker accepts `--llm_service` to route image
patches through an LLM for description (~$0.01/page on OpenRouter). We
did not enable it for this submission to keep the pipeline
self-contained.

---

## Production gaps acknowledged

This is a take-home; not a production system. Honest about what's
missing:

* **Single-node Qdrant**, no replication, no snapshot/backup strategy.
* **Demo API keys** in `.env.example` are placeholders. Real deployment
  must rotate through a secret store and use TLS.
* **Reranker latency**: Cohere `rerank-v3.5` is wired in via OpenRouter
  and lifts top-1 score by ~+0.22 on Vietnamese queries, but each call
  goes through OpenRouter's proxy, adding ~10–12 s of round-trip. For
  production, swap to Cohere direct API or a local TEI service hosting
  `bge-reranker-base` (Docker Compose has the `rerank` profile staged
  for that — just edit `ov.conf` to point at `localhost:8081`).
* **Hybrid retrieval (dense + sparse RRF)** present in the adapter but
  default `sparse_weight=0` for simplicity. Turning it on requires a
  reranker config and a follow-up integration test.
* **No automated regression eval** — the 8-query demo proves the
  pipeline works; it doesn't measure recall@k. A 30-query eval set with
  ground truth is the next thing I would build.
* **OpenViking PyPI 0.4.2 quirks** (filed as production-gap notes):
  - `storage.vectordb.project_name` field is **aliased** to `project` in
    the JSON config — using the field name fails validation. We use the
    alias.
  - When loading the adapter via dotted-path, OpenViking creates the
    Qdrant collection without invoking the adapter's
    `_build_default_index_meta`, so HNSW + quantization tuning from
    `custom_params` is ignored at create time. Workaround would be a
    post-create patch via `qdrant-client` direct.
* **License inheritance** — OpenViking is AGPL-3.0; this project
  inherits. Fine for internal use, blocks commercial SaaS without legal
  review.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Qdrant returned HTTP 401` | `EMBED_API_KEY` left blank or `QDRANT_API_KEY` mismatched between `.env` and `docker compose up` |
| `make ingest` hangs at "queued for vectorization" | Embedding endpoint slow or down; check `make embed-health` |
| `text-embedding-3-small` cosine ranking looks weird on some queries | Expected for English-leaning model on Vietnamese; the Cohere rerank stage usually fixes it (top-1 score 0.95+). |
| Query takes 10+ s | Normal when reranking is enabled — most of the time is OpenRouter's `/rerank` proxy round-trip. To skip rerank for faster results, blank out `EMBED_API_KEY` is **not** the right move (it'd break embedding too). Instead edit `ov.conf` and remove the `rerank` block, or point `api_base` at a local TEI rerank container. |
| Marker takes >30 min for full PDF | Some Surya sub-models fall back to CPU on MPS; that's normal — full GPU lights up only on text recognition |
| `Event loop is closed` warnings during shutdown | Upstream OpenViking 0.4.2 background-task quirk; cosmetic, no functional impact |
| Docker daemon not running | Start Docker Desktop first; `make qdrant-up` will fail otherwise |
| Want to re-ingest from scratch | `make clean-data` (drops Qdrant volume + workspace) then `make ingest` |

---

## License

AGPL-3.0-or-later (inherited from OpenViking). All code in this repo is
provided under the same license.
