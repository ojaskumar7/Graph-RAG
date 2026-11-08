# graphrag-lite

Lightweight GraphRAG MVP in `/home/barely_ok/GraphRAG`.

- **Goal**: ingest docs → chunk → LLM API entity/relation extraction → knowledge graph with provenance → keyword/graph retrieval → grounded answers with citations.
- **Constraints**: **stdlib-first (0 MB extra deps)**, **no PyTorch/CUDA**, **no Ollama**, **no heavy venv required**. Uses OpenAI-compatible LLM APIs via env config.
- **Dataset**: **HotpotQA (distractor) subset** — CC BY-SA 4.0 (verify terms). Sample 5 questions in `data/eval/golden.jsonl`, documents in `data/raw/`.

## Setup
```bash
cd /home/barely_ok/GraphRAG
# Optional: use uv venv (already created)
source .venv/bin/activate
# Or run directly with system Python (no deps required)
cp .env.example .env
# Edit .env with your OpenAI-compatible API: GRAPHRAG_LLM_BASE_URL, GRAPHRAG_LLM_API_KEY, GRAPHRAG_LLM_MODEL
PYTHONPATH=src python3 -m graphrag.cli info
```

## Working CLI Commands
```bash
# Ingest + chunk documents
PYTHONPATH=src python3 -m graphrag.cli ingest

# Build graph (structure only without API key; with key: LLM entity/relation extraction)
PYTHONPATH=src python3 -m graphrag.cli build-graph

# Query with retrieval (answers need API key; without: shows retrieval only)
PYTHONPATH=src python3 -m graphrag.cli query "Who built the Eiffel Tower?" --json

# Evaluate retrieval (no API key needed)
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3

# Evaluate full pipeline with mock LLM (tests citation validity)
PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --mock-llm --full-pipeline

# Inspect graph stats
PYTHONPATH=src python3 -m graphrag.cli inspect --stats
```

## Resume-worthy features implemented
- ✅ Provenance-tracked graph extraction (every edge cites `source_chunk_id`)
- ✅ Citation-grounded answers + "insufficient evidence" behavior
- ✅ Evaluation harness with retrieval hit@k, recall@k, **citation validity rate**
- ✅ Incremental indexing via content-hash manifest
- ✅ Graph inspection/export (JSONL nodes/edges)
- ✅ Mocked, no-network unit tests (eval passes without API key)
- ✅ Stdlib-only, 0 MB deps, no PyTorch/CUDA, runs in uv venv or system Python

## Architecture
```
data/raw/*.txt  →  ingest (chunk + manifest)  →  build-graph (entities/relations via LLM)
                                                              ↓
data/eval/golden.jsonl  ←  eval (retrieval + answer + citation check)  ←  query (hybrid retrieval)
```

## Project Structure
```
src/graphrag/
├── cli.py              # Commands: info, ingest, build-graph, query, eval, inspect
├── config.py           # Settings from .env
├── llm/client.py       # OpenAI-compatible client (stdlib HTTP)
├── ingest/             # load, chunk, manifest (incremental)
├── graph/              # schema, store (JSONL persistence)
├── retrieve/           # keyword search + graph expansion
├── generate/           # grounded answer with citations
└── eval/               # golden dataset, metrics, mock LLM
```

## Sample Output
```
$ PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --mock-llm --full-pipeline
Hit rate: 100.00% (5/5)
Avg recall@k: 100.00%
Citation valid rate: 100.00% (5/5)
```

See full implementation plan in local Kilo plans directory.