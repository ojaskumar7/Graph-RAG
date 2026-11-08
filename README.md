# graphrag-lite

![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/deps-stdlib--only-00bfff)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Lightweight **GraphRAG (Graph Retrieval-Augmented Generation)** MVP implemented in pure Python standard library — **zero external dependencies**, no PyTorch/CUDA, no heavy virtual environments.

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

## 🏗 Architecture

```mermaid
flowchart TD
    A[data/raw/*.txt] --> B[ingest: chunk + manifest]
    B --> C[build-graph: LLM extraction]
    C --> D[(Graph Store: JSONL)]
    D --> E[build-fts: SQLite FTS5 index]
    F[query: "question"] --> G[retrieve: FTS5 + graph expansion]
    G --> H[generate: grounded answer + citations]
    G --> D
    I[data/eval/golden.jsonl] --> J[eval: hit@k / recall@k / citation validity]
    J --> D
    K[inspect: stats / export JSON / DOT] --> D
```

---

## 🚀 Quick Start

```bash
cd /home/barely_ok/GraphRAG

# 1. Configure LLM API (optional — works without for retrieval/eval)
cp .env.example .env
# Edit .env with your OpenAI-compatible endpoint:
# GRAPHRAG_LLM_BASE_URL=https://api.openai.com/v1
# GRAPHRAG_LLM_API_KEY=sk-...
# GRAPHRAG_LLM_MODEL=gpt-4o-mini

# 2. Ingest documents
PYTHONPATH=src python3 -m graphrag.cli ingest
# Output: Ingested 2 docs -> 3 chunks (manifest written)

# 3. Build knowledge graph (LLM extraction requires API key)
PYTHONPATH=src python3 -m graphrag.cli build-graph
# Output: Graph built: {'Document': 2, 'Chunk': 3, 'Entity': 8, 'RELATED_TO': 12} nodes, ...

# 4. Build FTS5 search index
PYTHONPATH=src python3 -m graphrag.cli build-fts

# 5. Query with citations
PYTHONPATH=src python3 -m graphrag.cli query "Who built the Eiffel Tower?" --json

# 6. Evaluate pipeline
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --mock-llm --full-pipeline
```

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
| `inspect` | Graph stats / export JSON/DOT | `--stats`, `--export-json`, `--export-dot`, `--subgraph-chunks`, `--hops`, `--max-nodes` |

### Examples

```bash
# Query with custom retrieval params
PYTHONPATH=src python3 -m graphrag.cli query "Where is the Eiffel Tower?" --top-k 5 --hops 2 --json

# Eval with detailed output
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 5 --hops 2 --output report.json --full-pipeline

# Export full graph as JSON
PYTHONPATH=src python3 -m graphrag.cli inspect --export-json graph.json

# Export subgraph for specific chunks as DOT (visualize with Graphviz)
PYTHONPATH=src python3 -m graphrag.cli inspect --subgraph-chunks eeb24a88f8ee5d4d --export-dot subgraph.dot --hops 2

# Graphviz rendering
dot -Tpng subgraph.dot -o subgraph.png
```

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHRAG_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `GRAPHRAG_LLM_API_KEY` | *(required for LLM)* | API key |
| `GRAPHRAG_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `GRAPHRAG_DATA_DIR` | `./data` | Data root directory |
| `GRAPHRAG_REQUEST_TIMEOUT` | `60` | HTTP timeout (seconds) |
| `GRAPHRAG_MAX_RETRIES` | `2` | LLM retry attempts |

---

## 🧪 Testing (No API Key Required)

```bash
# Run extraction unit tests (10 tests covering JSON repair, validation, retries)
PYTHONPATH=src python3 tests/test_extract.py

# Run eval with mock LLM (validates retrieval + citation logic)
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --mock-llm --full-pipeline

# Expected output:
# Hit rate: 100.00% (5/5)
# Avg recall@k: 100.00%
# Citation valid rate: 100.00% (5/5)
```

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hit Rate** | % questions with ≥1 supporting chunk retrieved | 100% |
| **Recall@k** | Fraction of supporting facts found in top-k | 100% |
| **Citation Valid Rate** | % answers where all citations resolve to retrieved chunks | 100% |
| **Confidence** | Heuristic: 0.8 (cited), 0.4 (uncited), 0.1 (insufficient) | — |

---

## 📁 Project Structure

```
graphrag-lite/
├── .env.example              # API config template
├── .gitignore
├── pyproject.toml            # Package config (no deps)
├── README.md
├── data/
│   ├── raw/                  # Source documents (.txt/.md)
│   │   ├── eiffel_tower.txt
│   │   └── statue_of_liberty.txt
│   ├── eval/
│   │   ├── golden.jsonl      # 5 HotpotQA-style Q/A with supporting facts
│   │   └── README.md
│   └── processed/            # Generated artifacts (gitignored)
│       ├── manifest.jsonl    # Chunk content hashes
│       ├── nodes.jsonl       # Graph nodes
│       ├── edges.jsonl       # Graph edges
│       └── fts_index.sqlite  # FTS5 search index
├── src/graphrag/
│   ├── __init__.py
│   ├── cli.py                # All CLI commands
│   ├── config.py             # Settings from .env
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py         # OpenAI-compatible client (stdlib urllib)
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── loader.py         # Document loading
│   │   ├── chunk.py          # Sliding-window chunking
│   │   └── manifest.py       # Content-hash manifest (incremental)
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── pipeline.py       # ExtractionPipeline (retries, repair, validation)
│   │   └── normalize.py      # EntityNormalizer (dedup/canonicalization)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py         # Node/Edge types + factories
│   │   ├── store.py          # GraphStore (JSONL persistence)
│   │   └── export.py         # JSON/DOT export (full graph + subgraphs)
│   ├── retrieve/
│   │   ├── __init__.py
│   │   └── search.py         # FTS5 + keyword fallback + graph expansion
│   ├── generate/
│   │   ├── __init__.py
│   │   └── answer.py         # Grounded answer with citations
│   └── eval/
│       ├── __init__.py
│       └── metrics.py        # Evaluation harness + MockLLMClient
└── tests/
    ├── __init__.py
    ├── test_cfg.py
    ├── test_eval.py
    └── test_extract.py       # 10 extraction pipeline tests
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Stdlib-only** | Zero external deps → minimal venv size, no PyTorch/CUDA conflicts, works on any Python 3.12+ |
| **OpenAI-compatible API** | Works with OpenAI, OpenRouter, local gateways (vLLM, Ollama + OpenAI shim), self-hosted |
| **JSONL graph storage** | Human-readable, diffable, no DB server needed |
| **SQLite FTS5** | Built-in full-text search (BM25), zero extra deps |
| **Provenance on every edge** | Every `RELATED_TO` edge stores `source_chunk_id` → verifiable citations |
| **Mock LLM for eval** | CI-friendly, zero cost, deterministic |
| **Content-hash manifest** | Incremental indexing — unchanged chunks skip LLM calls |

---

## 📈 Resume Highlights

- **Provenance-tracked GraphRAG**: Every extracted relation cites its source chunk; answers cite evidence chunks
- **Evaluation Harness**: Automated hit@k, recall@k, citation validity rate with JSON reports
- **Robust LLM Extraction**: Multi-strategy JSON repair (code blocks, regex, common fixes), schema validation, configurable retries
- **Incremental Indexing**: Content-hash manifest avoids re-processing unchanged documents
- **Graph Export**: JSON + DOT (Graphviz) for full graph and query-specific subgraphs
- **Zero-Dependency**: Pure stdlib implementation — no PyTorch, no CUDA, no heavy venv
- **Testable**: Mocked LLM client enables full pipeline testing without API keys or network

---

## 📄 Sample Output

### Query with Citations
```json
{
  "question": "Who built the Eiffel Tower?",
  "answer": "The Eiffel Tower was built by Gustave Eiffel's company [eeb24a88f8ee5d4d].",
  "citations": ["eeb24a88f8ee5d4d"],
  "confidence": 0.8,
  "retrieval": {
    "keyword_chunks": 2,
    "query_entities": ["Eiffel Tower", "Gustave Eiffel"],
    "expanded_entity_count": 3,
    "total_chunks_retrieved": 3
  }
}
```

### Evaluation Report
```json
{
  "total_questions": 5,
  "hit_at_k": 5,
  "hit_rate": 1.0,
  "avg_recall_at_k": 1.0,
  "citations_valid_count": 5,
  "citation_valid_rate": 1.0,
  "details": [...]
}
```

---

## 🔮 Future Enhancements

- [ ] Local embeddings (ONNX / fastembed) for semantic retrieval
- [ ] Community detection / graph summarization (MS GraphRAG style)
- [ ] PDF/HTML ingestion via `pdfplumber` / `beautifulsoup4`
- [ ] FastAPI server for `/query` endpoint
- [ ] Larger HotpotQA / WikiText subsets
- [ ] Mermaid architecture diagram in README (already here ✅)

---

## 📜 License

MIT License — see `LICENSE` (add one if needed).

---

## 🙏 Credits

- HotpotQA dataset (CC BY-SA 4.0) — https://hotpotqa.github.io/
- Inspired by Microsoft GraphRAG and llama-index GraphRAG patterns
- Built with Python stdlib + uv for virtual environment management