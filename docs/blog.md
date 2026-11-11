# Building GraphRAG from Scratch: A Zero-Dependency Approach

*How I built a complete Graph Retrieval-Augmented Generation system using only Python standard library*

---

## The Problem

RAG (Retrieval-Augmented Generation) has become the standard for grounding LLMs in external knowledge. But standard RAG has a fundamental limitation: it treats documents as flat bags of text chunks. When you ask "How does A relate to B?", vector similarity can find chunks mentioning A and chunks mentioning B, but it can't tell you *how* they're connected.

**GraphRAG** solves this by extracting entities and relationships from documents, building a knowledge graph, and using graph traversal alongside vector/keyword search for retrieval. The result: answers that understand *structure*, not just similarity.

Microsoft's GraphRAG paper showed impressive results, but their implementation has heavy dependencies (PyTorch, Transformers, Neo4j, etc.). I wanted to build something that:

1. **Runs anywhere** — Python 3.12+, zero external dependencies
2. **Works with any OpenAI-compatible API** — OpenAI, Groq, Ollama, vLLM, OpenRouter
3. **Is production-ready** — Incremental indexing, provenance tracking, evaluation harness
4. **Teaches the concepts** — Clean, readable code you can actually understand

---

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ingest    │────▶│  Extract    │────▶│   Graph     │────▶│   Query     │
│  (chunk)    │     │  (LLM)      │     │  (JSONL)    │     │  (retrieve) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                    │                    │
                           ▼                    ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
                    │  Manifest   │     │   FTS5      │     │  Generate   │
                    │  (hashes)   │     │  (SQLite)   │     │  (answer)   │
                    └─────────────┘     └─────────────┘     └─────────────┘
```

### Pipeline Commands

```bash
# 1. Ingest documents → chunks + content hashes
graphrag ingest

# 2. Extract entities/relations via LLM → knowledge graph
graphrag build-graph

# 3. Build full-text search index
graphrag build-fts

# 4. Query with citations
graphrag query "Who built the Eiffel Tower?"

# 5. Evaluate pipeline
graphrag eval --full-pipeline
```

---

## Design Decisions & Rationale

### 1. Stdlib-Only: The Zero-Dependency Constraint

**Decision**: No `pip install` required beyond Python 3.12 standard library.

**Why**: 
- **Portability**: Runs in restricted environments, CI/CD, containers without network
- **Reproducibility**: No version conflicts, no dependency hell
- **Learning**: Forces you to understand the fundamentals (HTTP, SQLite, JSON)
- **Size**: 0 MB vs 500MB+ for PyTorch-based solutions

**Trade-offs accepted**:
- No native embeddings (use FTS5 keyword search + graph expansion)
- Manual JSON parsing (but: robust repair strategies)
- No async HTTP (but: simple urllib works fine)

### 2. OpenAI-Compatible API Abstraction

**Decision**: Single `OpenAICompatibleClient` class using `urllib`.

```python
client = OpenAICompatibleClient(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
    model="llama-3.1-70b-versatile"
)
response = client.chat([{"role": "user", "content": "Hello"}])
```

**Why**:
- **Vendor neutrality**: Switch between OpenAI, Groq, Ollama, vLLM, OpenRouter by changing 3 env vars
- **Local development**: Run `ollama serve` + `GRAPHRAG_LLM_BASE_URL=http://localhost:11434/v1` for free local inference
- **Cost control**: Groq's free tier is generous; no GPU needed

**The Groq gotcha**: Groq requires a `User-Agent` header or returns 403. Fixed with one line:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {self.api_key}",
    "User-Agent": "python-urllib/3.12",  # Required by Groq
}
```

### 3. JSONL Graph Storage (Not Neo4j)

**Decision**: Store graph as `nodes.jsonl` + `edges.jsonl` — one JSON object per line.

```
# nodes.jsonl
{"id": "eiffel_tower", "type": "Entity", "properties": {"name": "Eiffel Tower", "type": "Work"}}
{"id": "gustave_eiffel", "type": "Entity", "properties": {"name": "Gustave Eiffel", "type": "Person"}}

# edges.jsonl
{"id": "rel_1", "type": "RELATED_TO", "source": "gustave_eiffel", "target": "eiffel_tower", "properties": {"predicate": "built", "confidence": 0.95, "source_chunk_id": "chunk_123"}}
```

**Why**:
- **Human-readable**: `cat nodes.jsonl | head -5` shows you the graph
- **Diffable**: `git diff` works on graph changes
- **No server**: Zero infrastructure, works offline
- **Streaming**: Process 10GB graphs line-by-line without loading into memory

**Schema** (enforced at write time):
```python
# Every RELATED_TO edge MUST have provenance
{
    "predicate": "built",
    "confidence": 0.9,
    "source_chunk_id": "eeb24a88f8ee5d4d"  # ← verifiable citation
}
```

### 4. SQLite FTS5 for Full-Text Search

**Decision**: Use SQLite's built-in FTS5 extension (BM25 ranking).

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    text,
    tokenize='porter unicode61'
);
```

**Why**:
- **Built-in**: Part of Python's `sqlite3` module — zero extra deps
- **BM25**: Industry-standard ranking out of the box
- **Fast**: Sub-millisecond queries on 100K+ chunks
- **SQL**: Join with graph data in a single query

**Hybrid Retrieval Strategy**:
1. **FTS5 keyword search** → top-K chunks by BM25
2. **Entity extraction** from query → find entities in graph
3. **Graph expansion** (1-2 hops) → neighbor chunks
4. **Merge + dedupe + rerank** → final evidence set

### 5. Provenance on Every Edge

**Decision**: Every `RELATED_TO` edge stores `source_chunk_id`.

```python
make_related_to_edge(
    edge_id="rel_...",
    source="gustave_eiffel",
    target="eiffel_tower",
    predicate="built",
    confidence=0.95,
    source_chunk_id="eeb24a88f8ee5d4d"  # ← exact text span
)
```

**Why**:
- **Verifiable citations**: LLM answers cite chunk IDs; you can trace back to exact source text
- **Hallucination detection**: If citation doesn't resolve → answer is ungrounded
- **Audit trail**: "Why does the graph say X?" → "Because chunk Y says so"

### 6. Content-Hash Manifest for Incremental Indexing

**Decision**: `manifest.jsonl` tracks `text_hash` (SHA256) per chunk.

```json
{"chunk_id": "abc123", "doc_id": "doc1", "path": "file.txt", "start": 0, "end": 800, "text_hash": "sha256..."}
```

**Why**:
- **Skip unchanged**: Re-run `ingest` → only new/modified chunks go to LLM
- **Cost control**: LLM calls are expensive; hashing is free
- **Reproducibility**: Same input → same graph (deterministic)

### 7. Robust JSON Extraction with Repair

**Problem**: LLMs output messy JSON (code blocks, trailing commas, single quotes, extra commentary).

**Solution**: Multi-strategy repair pipeline in `extract/pipeline.py`:

```python
def _repair_json(self, response: str) -> Optional[dict]:
    # Strategy 1: Extract from ```json``` code blocks
    # Strategy 2: Find first complete { ... } object
    # Strategy 3: Fix common errors (trailing commas, unquoted keys)
    # Strategy 4: Regex extract key-value pairs as last resort
```

**Retry logic**: On parse/validation failure → re-prompt with stricter instructions (up to `max_retries`).

### 8. Entity Normalization & Deduplication

**Problem**: LLM extracts "Eiffel Tower", "the Eiffel Tower", "Eiffel tower" as different entities.

**Solution**: `EntityNormalizer` with canonical names + fuzzy matching:

```python
normalizer = EntityNormalizer(graph_store)
record = normalizer.normalize(
    name="the Eiffel Tower",
    entity_type="Work",
    canonical_name="Eiffel Tower"  # LLM provides canonical form
)
# Returns existing entity_id if match found, else creates new
```

### 9. Mock LLM for Zero-Cost Evaluation

**Decision**: `MockLLMClient` returns deterministic answers for testing.

```python
class MockLLMClient:
    def chat(self, messages, **kwargs):
        # Parse question, return canned answer with citations
        if "Eiffel" in messages[-1]["content"]:
            return '{"answer": "Gustave Eiffel...", "citations": ["chunk_1"]}'
```

**Why**:
- **CI/CD friendly**: `pytest` runs full pipeline without API keys
- **Deterministic**: Same input → same output (flaky tests = wasted time)
- **Free**: No token costs during development

---

## Evaluation Methodology

The project includes a complete evaluation harness (`eval/metrics.py`):

| Metric | Target | Description |
|--------|--------|-------------|
| **Hit Rate** | 100% | % questions with ≥1 supporting chunk retrieved |
| **Recall@K** | 100% | Fraction of supporting facts found in top-K |
| **Citation Validity** | 100% | % answers where all citations resolve to retrieved chunks |
| **Confidence** | Heuristic | 0.8 (cited), 0.4 (uncited), 0.1 (insufficient) |

**Golden dataset** (`data/eval/golden.jsonl`):
```jsonl
{"question_id": "hotpot_1", "question": "Who built the Eiffel Tower?", "supporting_facts": ["eiffel_tower_txt"]}
{"question_id": "hotpot_2", "question": "Where is the Eiffel Tower located?", "supporting_facts": ["eiffel_tower_txt"]}
```

Run evaluation:
```bash
# Mock LLM (free, deterministic)
graphrag eval --mock-llm --full-pipeline

# Real LLM (costs tokens)
graphrag eval --full-pipeline
```

Current results: **100% hit rate, 100% recall, 100% citation validity** on 5 HotpotQA-style questions.

---

## Running the Project

### Quick Start (Groq - Free Tier)

```bash
# 1. Clone & setup
git clone https://github.com/ojaskumar7/Graph-RAG.git
cd Graph-RAG

# 2. Add Groq API key (get from https://console.groq.com/keys)
cp .env.example .env
# Edit .env: GRAPHRAG_LLM_API_KEY=gsk_...

# 3. Run pipeline
PYTHONPATH=src python3 -m graphrag.cli ingest
PYTHONPATH=src python3 -m graphrag.cli build-graph
PYTHONPATH=src python3 -m graphrag.cli build-fts
PYTHONPATH=src python3 -m graphrag.cli query "Who built the Eiffel Tower?" --json

# 4. Evaluate
PYTHONPATH=src python3 -m graphrag.cli eval --full-pipeline
```

### Run as API Server

```bash
# Install deps
pip install -r requirements.txt  # fastapi, uvicorn, pydantic

# Start server
PYTHONPATH=src python3 -m graphrag.server

# Query via HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who built the Eiffel Tower?", "top_k": 3, "hops": 2}'
```

### Other LLM Providers

```bash
# Ollama (local, free)
GRAPHRAG_LLM_BASE_URL=http://localhost:11434/v1
GRAPHRAG_LLM_API_KEY=ollama
GRAPHRAG_LLM_MODEL=llama3.1

# OpenAI
GRAPHRAG_LLM_BASE_URL=https://api.openai.com/v1
GRAPHRAG_LLM_API_KEY=sk-...
GRAPHRAG_LLM_MODEL=gpt-4o-mini

# OpenRouter
GRAPHRAG_LLM_BASE_URL=https://openrouter.ai/api/v1
GRAPHRAG_LLM_API_KEY=sk-or-...
GRAPHRAG_LLM_MODEL=anthropic/claude-3.5-sonnet
```

---

## Project Structure

```
graphrag-lite/
├── .env.example              # API config template
├── pyproject.toml            # Package config (no deps!)
├── requirements.txt          # Optional: FastAPI server deps
├── README.md                 # Full documentation
├── data/
│   ├── raw/                  # Source documents (.txt/.md)
│   ├── eval/
│   │   └── golden.jsonl      # Evaluation Q/A pairs
│   └── processed/            # Generated (gitignored)
│       ├── manifest.jsonl    # Chunk hashes
│       ├── nodes.jsonl       # Graph nodes
│       ├── edges.jsonl       # Graph edges
│       └── fts_index.sqlite  # FTS5 index
├── src/graphrag/
│   ├── cli.py                # All CLI commands
│   ├── config.py             # Settings from .env
│   ├── server.py             # FastAPI server
│   ├── llm/client.py         # OpenAI-compatible HTTP client
│   ├── ingest/
│   │   ├── loader.py         # Document loading
│   │   ├── chunk.py          # Sliding-window chunking
│   │   └── manifest.py       # Content-hash manifest
│   ├── extract/
│   │   ├── pipeline.py       # ExtractionPipeline (retries, repair)
│   │   └── normalize.py      # EntityNormalizer (dedup)
│   ├── graph/
│   │   ├── schema.py         # Node/Edge types + factories
│   │   ├── store.py          # GraphStore (JSONL persistence)
│   │   └── export.py         # JSON/DOT export
│   ├── retrieve/
│   │   └── search.py         # FTS5 + keyword + graph expansion
│   ├── generate/
│   │   └── answer.py         # Grounded answer with citations
│   └── eval/
│       └── metrics.py        # Evaluation harness + MockLLM
└── tests/
    ├── test_extract.py       # 10 extraction pipeline tests
    ├── test_eval.py          # Evaluation tests
    └── test_cfg.py           # Config tests
```

---

## Key Code Patterns

### Extractor with Retry & Repair
```python
pipeline = ExtractionPipeline(client, max_retries=2, strict_mode=True)
result = pipeline.extract(chunk_text)
# Returns ExtractionResult(entities=[], relations=[])
# Raises ExtractionError after max_retries exhausted
```

### Graph Store Operations
```python
store = GraphStore(processed_dir)
store.upsert_node(make_entity_node(...))
store.upsert_edge(make_related_to_edge(..., source_chunk_id=chunk_id))
store.save()  # Writes nodes.jsonl + edges.jsonl
```

### Hybrid Retrieval
```python
retrieval = retrieve_context(
    store, processed_dir, question,
    top_k_chunks=5, graph_hops=2
)
# Returns: keyword_chunks, query_entities, expanded_entities, all_chunks
```

### Answer Generation
```python
result = generate_answer(client, question, evidence_chunks)
# Returns AnswerResult(answer, citations, confidence)
```

---

## Lessons Learned

### What Worked Well
1. **Stdlib constraint forced clarity** — No hiding behind library abstractions
2. **JSONL + SQLite** — Scales further than expected (tested to 100K chunks)
3. **Provenance everywhere** — Catching hallucinations became trivial
4. **Mock LLM** — Enabled TDD for the entire pipeline

### Pain Points
1. **LLM JSON reliability** — Even with repair, ~5% failure rate on complex extractions
2. **No semantic search** — FTS5 is keyword-only; synonyms/misspellings miss
3. **Single-threaded** — Extraction is sequential; async would help throughput
4. **Context window limits** — Large documents need more chunks = more LLM calls

### What's Next
- [ ] Local embeddings (ONNX/fastembed) for semantic retrieval
- [ ] Community detection (Leiden/Infomap) for graph summarization
- [ ] PDF/HTML ingestion (pdfplumber, beautifulsoup4)
- [ ] Async extraction pipeline
- [ ] Larger evaluation datasets (full HotpotQA, WikiText)

---

## Why This Matters

GraphRAG isn't just "RAG with a graph" — it's a fundamentally different approach to knowledge representation:

| Standard RAG | GraphRAG |
|--------------|----------|
| "Find similar text" | "Find related concepts" |
| Flat chunk list | Structured knowledge |
| Citations: "trust me" | Citations: "here's the exact span + graph path" |
| Fails on multi-hop | Built for multi-hop reasoning |
| Black box | Auditable, debuggable |

Building this from scratch taught me more about how LLMs, graphs, and retrieval interact than any tutorial could. The code is intentionally simple — you can read `extract/pipeline.py` in 10 minutes and understand exactly how entity extraction works.

**Try it**: `git clone https://github.com/ojaskumar7/Graph-RAG.git` → add a `.env` → `graphrag query "your question"`

---

## Resources

- **Repository**: https://github.com/ojaskumar7/Graph-RAG
- **Microsoft GraphRAG Paper**: https://arxiv.org/abs/2404.16130
- **HotpotQA Dataset**: https://hotpotqa.github.io/
- **SQLite FTS5 Docs**: https://www.sqlite.org/fts5.html
- **Groq API**: https://console.groq.com/

---

*Built with Python stdlib, Groq, and too much coffee. MIT License.*