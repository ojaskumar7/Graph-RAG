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

### System Overview

```mermaid
flowchart TD
    %% External
    LLM_API[("LLM API\nOpenAI-compatible")]
    USER[("User")]

    %% Data Layer
    subgraph DATA["Data Layer"]
        RAW[(data/raw/*.txt\n.md/.txt files)]
        GOLDEN[(data/eval/golden.jsonl\nGolden Q/A)]
        PROCESSED[("data/processed/")]
        MANIFEST[(manifest.jsonl\nContent hashes)]
        NODES[(nodes.jsonl\nGraph nodes)]
        EDGES[(edges.jsonl\nGraph edges)]
        FTS[(fts_index.sqlite\nFTS5 index)]
    end

    %% Pipeline Commands
    subgraph PIPELINE["Pipeline Commands"]
        CMD_INGEST["ingest\nLoad → Chunk → Manifest"]
        CMD_BUILD_GRAPH["build-graph\nLLM Extract → Entities/Relations"]
        CMD_BUILD_FTS["build-fts\nSQLite FTS5 Index"]
        CMD_QUERY["query\nRetrieve → Generate Answer"]
        CMD_EVAL["eval\nRetrieval + Citation Validation"]
        CMD_INSPECT["inspect\nStats / Export JSON / DOT"]
    end

    %% Core Modules
    subgraph CORE["Core Modules"]
        INGEST_MOD["ingest/\nloader.py, chunk.py, manifest.py"]
        EXTRACT_MOD["extract/\npipeline.py, normalize.py"]
        GRAPH_MOD["graph/\nschema.py, store.py, export.py"]
        RETRIEVE_MOD["retrieve/\nsearch.py"]
        GENERATE_MOD["generate/\nanswer.py"]
        EVAL_MOD["eval/\nmetrics.py"]
        LLM_MOD["llm/client.py\nOpenAI-compatible HTTP"]
    end

    %% Flow
    USER -->|Configure| LLM_API
    USER -->|Run commands| CMD_INGEST
    USER -->|Run commands| CMD_BUILD_GRAPH
    USER -->|Run commands| CMD_BUILD_FTS
    USER -->|Run commands| CMD_QUERY
    USER -->|Run commands| CMD_EVAL
    USER -->|Run commands| CMD_INSPECT

    %% Ingest flow
    RAW -->|Read files| INGEST_MOD
    INGEST_MOD -->|Chunk + Hash| MANIFEST
    INGEST_MOD -->|Chunk text| PROCESSED

    %% Build graph flow
    MANIFEST -->|Load chunks| EXTRACT_MOD
    LLM_API -->|Extract entities/relations| EXTRACT_MOD
    EXTRACT_MOD -->|Persist nodes| NODES
    EXTRACT_MOD -->|Persist edges| EDGES
    EXTRACT_MOD -->|Provenance| MANIFEST

    %% FTS flow
    NODES -->|Chunk text| CMD_BUILD_FTS
    CMD_BUILD_FTS -->|Index| FTS

    %% Query flow
    CMD_QUERY -->|Question| RETRIEVE_MOD
    RETRIEVE_MOD -->|FTS5 search| FTS
    RETRIEVE_MOD -->|Graph expansion| EDGES
    RETRIEVE_MOD -->|Chunks + Entities| GENERATE_MOD
    LLM_API -->|Grounded answer| GENERATE_MOD
    GENERATE_MOD -->|Answer + Citations| USER

    %% Eval flow
    GOLDEN -->|Golden Q/A| EVAL_MOD
    EVAL_MOD -->|Retrieve| RETRIEVE_MOD
    EVAL_MOD -->|Mock/Real LLM| LLM_API
    EVAL_MOD -->|Metrics report| USER

    %% Inspect flow
    CMD_INSPECT -->|Load graph| GRAPH_MOD
    GRAPH_MOD -->|JSON/DOT export| USER
    GRAPH_MOD -->|Stats| USER

    %% Styling
    classDef data fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef cmd fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef ext fill:#fce4ec,stroke:#c2185b,stroke-width:2px;

    class RAW,GOLDEN,MANIFEST,NODES,EDGES,FTS,PROCESSED data;
    class CMD_INGEST,CMD_BUILD_GRAPH,CMD_BUILD_FTS,CMD_QUERY,CMD_EVAL,CMD_INSPECT cmd;
    class INGEST_MOD,EXTRACT_MOD,GRAPH_MOD,RETRIEVE_MOD,GENERATE_MOD,EVAL_MOD,LLM_MOD core;
    class LLM_API,USER ext;
```

### Data Flow Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as graphrag.cli
    participant Ingest as ingest/
    participant Manifest as manifest.jsonl
    participant Extract as extract/
    participant LLM as LLM API
    participant Graph as graph/ (JSONL)
    participant FTS as fts_index.sqlite
    participant Retrieve as retrieve/
    participant Generate as generate/
    participant Eval as eval/

    User->>CLI: ingest
    CLI->>Ingest: Load .md/.txt
    Ingest->>Ingest: Chunk (800 chars, 100 overlap)
    Ingest->>Manifest: Write content hashes
    CLI-->>User: "Ingested N docs → M chunks"

    User->>CLI: build-graph
    CLI->>Manifest: Load chunks
    loop For each chunk
        CLI->>Extract: Extract entities/relations
        Extract->>LLM: Prompt with chunk text
        LLM-->>Extract: JSON (entities + relations)
        Extract->>Extract: Validate + Repair JSON
        Extract->>Extract: Normalize/Dedup entities
        Extract->>Graph: Persist nodes + edges
    end
    CLI-->>User: "Graph built: X nodes, Y edges"

    User->>CLI: build-fts
    CLI->>Graph: Load chunk nodes
    CLI->>FTS: Create FTS5 index
    CLI-->>User: "FTS5 index built"

    User->>CLI: query "question"
    CLI->>Retrieve: Retrieve context
    Retrieve->>FTS: FTS5 BM25 search
    Retrieve->>Graph: Graph expansion (1-2 hops)
    Retrieve-->>CLI: Chunks + Entities
    CLI->>Generate: Generate answer
    Generate->>LLM: Prompt with evidence + citations
    LLM-->>Generate: Answer with [chunk_id] citations
    Generate-->>CLI: Answer + Citations + Confidence
    CLI-->>User: JSON/Text response

    User->>CLI: eval --mock-llm --full-pipeline
    CLI->>Eval: Load golden.jsonl
    loop For each question
        Eval->>Retrieve: Retrieve context
        Eval->>LLM: Mock/Real answer
        Eval->>Eval: Validate citations
        Eval->>Eval: Compute hit@k, recall@k, citation validity
    end
    CLI-->>User: Evaluation report
```

### Graph Schema (ER Diagram)

```mermaid
erDiagram
    DOCUMENT ||--o{ CHUNK : "HAS_CHUNK"
    CHUNK ||--o{ ENTITY : "MENTIONS"
    ENTITY }|--o{ ENTITY : "RELATED_TO"
    
    DOCUMENT {
        string id PK
        string path
    }
    
    CHUNK {
        string id PK
        string doc_id FK
        string text
        int start
        int end
        string text_hash
    }
    
    ENTITY {
        string id PK
        string name
        string canonical_name
        string entity_type
    }
    
    RELATED_TO {
        string id PK
        string source FK
        string target FK
        string predicate
        float confidence
        string source_chunk_id FK
    }
    
    MENTIONS {
        string id PK
        string chunk_id FK
        string entity_id FK
    }
```

### Hybrid Retrieval Flow

```mermaid
graph LR
    subgraph Retrieval["Hybrid Retrieval"]
        FTS5["FTS5 BM25\nKeyword Search"]
        GRAPH["Graph Expansion\n1-2 hops"]
        MERGE["Merge + Dedupe\nScore + Rank"]
    end
    
    QUERY["User Question"] --> FTS5
    QUERY --> GRAPH
    FTS5 --> MERGE
    GRAPH --> MERGE
    MERGE --> EVIDENCE["Ranked Evidence Chunks"]
    EVIDENCE --> LLM["LLM Generation"]
    LLM --> ANSWER["Grounded Answer\n+ Citations"]
    
    classDef search fill:#e3f2fd,stroke:#1565c0;
    classDef merge fill:#fff3e0,stroke:#ef6c00;
    classDef gen fill:#e8f5e9,stroke:#2e7d32;
    
    class FTS5,GRAPH search;
    class MERGE merge;
    class LLM,ANSWER gen;
```

### Extraction Pipeline (with JSON Repair)

```mermaid
graph TD
    subgraph Extraction["Extraction Pipeline"]
        CHUNK_TEXT["Chunk Text"]
        PROMPT["Prompt Template\n+ Retry Prompt"]
        LLM_CALL["LLM API Call"]
        PARSE["JSON Parse"]
        REPAIR["JSON Repair\n• Code blocks\n• Regex extract\n• Trailing commas\n• Key quoting"]
        VALIDATE["Schema Validation\n• Required fields\n• Type checks\n• Confidence 0-1"]
        NORMALIZE["Entity Normalize\n• Canonical name\n• Dedup cache\n• Fuzzy match"]
        OUTPUT["ExtractionResult\nEntities + Relations"]
    end
    
    CHUNK_TEXT --> PROMPT
    PROMPT --> LLM_CALL
    LLM_CALL --> PARSE
    PARSE -->|Success| VALIDATE
    PARSE -->|Fail| REPAIR
    REPAIR -->|Fixed| VALIDATE
    REPAIR -->|Fail| LLM_CALL
    VALIDATE -->|Valid| NORMALIZE
    VALIDATE -->|Invalid| LLM_CALL
    NORMALIZE --> OUTPUT
    
    classDef step fill:#f3e5f5,stroke:#7b1fa2;
    classDef decision fill:#fff3e0,stroke:#ef6c00;
    
    class CHUNK_TEXT,PROMPT,LLM_CALL,PARSE,REPAIR,VALIDATE,NORMALIZE,OUTPUT step;
```

### Evaluation Harness

```mermaid
graph LR
    subgraph Evaluation["Evaluation Harness"]
        GOLDEN["golden.jsonl\nQuestions + Supporting Facts"]
        RETRIEVAL_EVAL["Retrieval Evaluation"]
        PIPELINE_EVAL["Full Pipeline Evaluation"]
        MOCK_LLM["MockLLMClient\nDeterministic Answers"]
        METRICS["Metrics:\n• Hit@k\n• Recall@k\n• Citation Valid Rate\n• Confidence"]
        REPORT["eval_report.json"]
    end
    
    GOLDEN --> RETRIEVAL_EVAL
    GOLDEN --> PIPELINE_EVAL
    PIPELINE_EVAL --> MOCK_LLM
    RETRIEVAL_EVAL --> METRICS
    PIPELINE_EVAL --> METRICS
    METRICS --> REPORT
    
    classDef eval fill:#e8f5e9,stroke:#2e7d32;
    class GOLDEN,RETRIEVAL_EVAL,PIPELINE_EVAL,MOCK_LLM,METRICS,REPORT eval;
```

> **📁 Full diagram source**: See [`docs/architecture.mmd`](docs/architecture.mmd) for the complete Mermaid source. Render locally with [Mermaid Live Editor](https://mermaid.live/) or VS Code Mermaid extension.

---

## 🚀 Quick Start

```bash
cd /home/barely_ok/GraphRAG

# 1. Configure LLM API (optional — works without for retrieval/eval)
cp .env.example .env
# Edit .env with your OpenAI-compatible endpoint:
# --- Groq (fast, free tier) ---
# GRAPHRAG_LLM_BASE_URL=https://api.groq.com/openai/v1
# GRAPHRAG_LLM_API_KEY=gsk_xxxxxxxxxxxx
# GRAPHRAG_LLM_MODEL=llama-3.1-70b-versatile
# --- OpenAI ---
# GRAPHRAG_LLM_BASE_URL=https://api.openai.com/v1
# GRAPHRAG_LLM_API_KEY=sk-...
# GRAPHRAG_LLM_MODEL=gpt-4o-mini
# --- Ollama (local) ---
# GRAPHRAG_LLM_BASE_URL=http://localhost:11434/v1
# GRAPHRAG_LLM_API_KEY=ollama
# GRAPHRAG_LLM_MODEL=llama3.1

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
| `GRAPHRAG_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL (Groq: `https://api.groq.com/openai/v1`, Ollama: `http://localhost:11434/v1`) |
| `GRAPHRAG_LLM_API_KEY` | *(required for LLM)* | API key (Groq: `gsk_...`, Ollama: `ollama`) |
| `GRAPHRAG_LLM_MODEL` | `gpt-4o-mini` | Model name (Groq: `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma2-9b-it`) |
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

## 📈 Actual Results (Groq: qwen/qwen3.8-27b)

| Metric | Mock LLM | Real LLM (Groq) |
|--------|----------|-----------------|
| **Hit Rate** | 100% (5/5) | 100% (5/5) |
| **Avg Recall@k** | 100% | 100% |
| **Citation Valid Rate** | 100% (5/5) | 100% (5/5) |
| **Graph Nodes** | N/A | 2 Document + 3 Chunk + 13 Entity = 18 |
| **Graph Edges** | N/A | 3 HAS_CHUNK + 17 MENTIONS + 18 RELATED_TO = 38 |

### Sample Answers (Real LLM)

| Question | Answer | Citations |
|----------|--------|-----------|
| Who built the Eiffel Tower? | Gustave Eiffel's company, Compagnie des Établissements Eiffel, designed and built the Eiffel Tower | `eeb24a88f8ee5d4d` |
| Where is the Eiffel Tower located? | The Eiffel Tower is located on the Champ de Mars in Paris, France | `eeb24a88f8ee5d4d` |
| Who designed the Statue of Liberty? | The Statue of Liberty was designed by French sculptor Frédéric Auguste Bartholdi | `dd72ce7db6bd2c56` |
| Who built the metal framework of the Statue of Liberty? | Gustave Eiffel | `dd72ce7db6bd2c56` |
| What was the 1889 World's Fair centerpiece? | The Eiffel Tower | `232d11df9ce35863` |

*All answers grounded with verifiable chunk citations. Confidence: 0.8 (cited).*

---

## 📁 Project Structure

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
| **OpenAI-compatible API** | Works with OpenAI, Groq, OpenRouter, local gateways (vLLM, Ollama + OpenAI shim), self-hosted |
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