# graphrag-lite

![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/deps-stdlib--only-00bfff)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Lightweight **GraphRAG (Graph Retrieval-Augmented Generation)** MVP in pure Python stdlib — **zero external dependencies**, no PyTorch/CUDA, no heavy venv.

---

## 🎯 Goal

Build a complete GraphRAG pipeline that:
1. **Ingests** documents (`.md`, `.txt`)
2. **Chunks** with overlap + content-hash manifest for incremental indexing
3. **Extracts** entities & relations via LLM API (OpenAI-compatible) with provenance tracking
4. **Persists** a knowledge graph (JSONL) with full citation provenance
5. **Retrieves** via SQLite FTS5 full-text search + graph neighbor expansion
6. **Answers** with grounded citations — "insufficient evidence" when support is weak
7. **Evaluates** retrieval hit@k, recall@k, and citation validity rate

---

## 🚀 Quick Start

```bash
# 1. Configure LLM API (optional — works without for retrieval/eval)
cp .env.example .env
# Edit .env with your OpenAI-compatible endpoint:
# Groq:     GRAPHRAG_LLM_BASE_URL=https://api.groq.com/openai/v1
#           GRAPHRAG_LLM_API_KEY=gsk_...
#           GRAPHRAG_LLM_MODEL=llama-3.1-70b-versatile
# Ollama:   GRAPHRAG_LLM_BASE_URL=http://localhost:11434/v1
#           GRAPHRAG_LLM_API_KEY=ollama
#           GRAPHRAG_LLM_MODEL=llama3.1

# 2. Ingest documents
PYTHONPATH=src python3 -m graphrag.cli ingest

# 3. Build knowledge graph (requires LLM API key)
PYTHONPATH=src python3 -m graphrag.cli build-graph

# 4. Build FTS5 search index
PYTHONPATH=src python3 -m graphrag.cli build-fts

# 5. Query with citations
PYTHONPATH=src python3 -m graphrag.cli query "Who built the Eiffel Tower?" --json

# 6. Evaluate pipeline
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --mock-llm --full-pipeline
```

---

## 🏗 Architecture

**See [`docs/architecture.mmd`](docs/architecture.mmd) for full diagrams** (system overview, data flow sequence, ER diagram, hybrid retrieval, extraction pipeline, evaluation harness).

Key components:
- **Ingest** → chunk + content-hash manifest (`manifest.jsonl`)
- **Extract** → LLM entities/relations → JSONL graph (`nodes.jsonl`, `edges.jsonl`)
- **Retrieve** → FTS5 BM25 + graph expansion (1-2 hops)
- **Generate** → Grounded answer with `[chunk_id]` citations
- **Evaluate** → hit@k, recall@k, citation validity rate

---

## 📋 CLI Reference

| Command | Description | Key Options |
|---------|-------------|-------------|
| `info` | Show config & project status | `--env-file` |
| `ingest` | Load & chunk `.md`/`.txt` from `data/raw/` | `--env-file` |
| `build-graph` | Extract entities/relations via LLM | `--env-file` |
| `build-fts` | Build SQLite FTS5 full-text index | `--env-file` |
| `query` | Retrieve + answer with citations | `question`, `--top-k`, `--hops`, `--json` |
| `eval` | Run evaluation on golden dataset | `--top-k`, `--hops`, `--mock-llm`, `--full-pipeline`, `--output` |
| `inspect` | Graph stats / export JSON/DOT | `--stats`, `--export-json`, `--export-dot`, `--subgraph-chunks` |

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHRAG_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `GRAPHRAG_LLM_API_KEY` | *(required for LLM)* | API key (Groq: `gsk_...`, Ollama: `ollama`) |
| `GRAPHRAG_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `GRAPHRAG_DATA_DIR` | `./data` | Data root directory |
| `GRAPHRAG_REQUEST_TIMEOUT` | `60` | HTTP timeout (seconds) |
| `GRAPHRAG_MAX_RETRIES` | `2` | LLM retry attempts |

**Groq models**: `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`, `qwen/qwen3.8-27b`

---

## 🧪 Testing (No API Key Required)

```bash
# Extraction unit tests (10 tests: JSON repair, validation, retries)
PYTHONPATH=src python3 tests/test_extract.py

# Eval unit tests (doc-ID matching, dedupe, extractive mock)
PYTHONPATH=src python3 tests/test_eval.py

# Eval with mock LLM on a subset (validates retrieval + citation logic)
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 5 --mock-llm --full-pipeline --limit 5

# Expected: Hit rate 100%, Recall@5 90%, Citation valid 100%
```

---

## 📊 Evaluation Results

| Metric | HotpotQA (retrieval-only) | Mock LLM full pipeline (n=5) |
|--------|---------------------------|------------------------------|
| **Hit Rate@5** | 100% (20/20) | 100% (5/5) |
| **Avg Recall@5** | 97.5% | 90.0% |
| **Citation Valid Rate** | — | 100% (5/5) |

> Evaluated on a **1,977-doc HotpotQA corpus** (2,142 chunks, structure-only graph — no LLM extraction), 20 multi-hop questions. Retrieval is FTS5 + vector only; the mock LLM is a deterministic extractive answerer, so citation metrics validate wiring rather than answer quality.

---

## 📁 Project Structure

```
graphrag-lite/
├── .env.example              # API config template
├── .gitignore
├── pyproject.toml            # Package config (no deps)
├── requirements.txt          # Optional: FastAPI server deps
├── README.md
├── data/
│   ├── raw/                  # Source documents (.txt/.md)
│   ├── eval/golden.jsonl     # 5 HotpotQA-style Q/A pairs
│   └── processed/            # Generated (gitignored)
│       ├── manifest.jsonl    # Chunk content hashes
│       ├── nodes.jsonl       # Graph nodes
│       ├── edges.jsonl       # Graph edges
│       └── fts_index.sqlite  # FTS5 search index
├── src/graphrag/             # Core modules
│   ├── cli.py                # All CLI commands
│   ├── server.py             # FastAPI server
│   ├── config.py             # Settings from .env
│   ├── llm/client.py         # OpenAI-compatible HTTP client
│   ├── ingest/               # loader, chunk, manifest
│   ├── extract/              # pipeline, normalize
│   ├── graph/                # schema, store, export
│   ├── retrieve/             # search (FTS5 + graph)
│   ├── generate/             # answer with citations
│   └── eval/                 # metrics + MockLLM
└── tests/
    ├── test_extract.py       # 10 extraction tests
    ├── test_eval.py
    └── test_cfg.py
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Stdlib-only** | Zero deps → minimal venv, no PyTorch/CUDA, works on any Python 3.12+ |
| **OpenAI-compatible API** | Works with OpenAI, Groq, Ollama, vLLM, OpenRouter, self-hosted |
| **JSONL graph storage** | Human-readable, diffable, no DB server needed |
| **SQLite FTS5** | Built-in full-text search (BM25), zero extra deps |
| **Provenance on every edge** | Every `RELATED_TO` stores `source_chunk_id` → verifiable citations |
| **Mock LLM for eval** | CI-friendly, zero cost, deterministic |
| **Content-hash manifest** | Incremental indexing — unchanged chunks skip LLM calls |

---

## 📈 Resume Highlights

- **Provenance-tracked GraphRAG**: Every relation cites source chunk; answers cite evidence
- **Evaluation Harness**: Automated hit@k, recall@k, citation validity with JSON reports
- **Robust LLM Extraction**: Multi-strategy JSON repair, schema validation, configurable retries
- **Incremental Indexing**: Content-hash manifest avoids re-processing unchanged documents
- **Graph Export**: JSON + DOT (Graphviz) for full graph and query-specific subgraphs
- **Zero-Dependency**: Pure stdlib — no PyTorch, no CUDA, no heavy venv
- **Testable**: Mocked LLM enables full pipeline testing without API keys

---

## 🔮 Future Enhancements

- [x] Local embeddings (ONNX / fastembed) for semantic retrieval
- [ ] Community detection / graph summarization (MS GraphRAG style)
- [ ] PDF/HTML ingestion via `pdfplumber` / `beautifulsoup4`
- [ ] Larger HotpotQA / WikiText evaluation subsets

---

## ⚠️ Limitations

Current eval uses a **1,977-document HotpotQA subset** (2,142 chunks, 20 multi-hop questions, structure-only graph). This tests:
- ✅ Pipeline wiring at modest scale
- ✅ Doc-level retrieval (hit@k / recall@k vs supporting docs)
- ✅ Citation formatting + mock LLM determinism

But does **NOT** test:
- ❌ Retrieval at scale (100K+ chunks)
- ❌ Answer quality with a real LLM (only extractive mock)
- ❌ Entity resolution / coreference (extraction not run at this scale)
- ❌ Graph-expanded multi-hop retrieval (no entities extracted yet)

See [`docs/blog.md`](docs/blog.md) for detailed discussion.

---

## 📜 License

MIT License.

---

## 🙏 Credits

- HotpotQA dataset (CC BY-SA 4.0) — https://hotpotqa.github.io/
- Inspired by Microsoft GraphRAG and llama-index GraphRAG patterns
- Built with Python stdlib + uv