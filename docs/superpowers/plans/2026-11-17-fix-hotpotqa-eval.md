# Fix HotpotQA Evaluation Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the GraphRAG eval harness produce honest, non-zero HotpotQA numbers by fixing the doc-ID mismatch, rebuilding the missing indexes, and replacing the hardcoded mock LLM with a deterministic extractive mock.

**Architecture:** Retrieval-only eval works off the structure-only graph (Doc/Chunk nodes + HAS_CHUNK edges) that `build-graph` writes when no LLM API key is set — no extraction cost. Doc-level hit/recall is computed by matching golden fact IDs (`doc_818633`) against chunk `doc_id`s (`doc_818633_txt`). A new data dir (`data_hotpot/`) with a `raw` symlink points ingest at the 1,977-doc corpus without touching the original `data/` layout.

**Tech Stack:** Python 3.12+ stdlib, SQLite FTS5, fastembed (already installed, local ONNX embeddings). No new dependencies.

## Global Constraints

- Core stays stdlib-only; fastembed is already an optional extra — do not add new deps.
- Tests run standalone (no pytest installed): `PYTHONPATH=src python3 tests/test_eval.py` with a `__main__` runner, matching `tests/test_extract.py` style.
- All CLI commands that touch the new corpus need `GRAPHRAG_DATA_DIR=data_hotpot`.
- Do not commit `eval_report.json`, corpus files, or anything under `data_hotpot/`.
- Pre-existing uncommitted changes (config.py, llm/client.py, extract/pipeline.py, metrics.py, golden.jsonl) are part of this work; they get committed with this effort. Confirm with user at the commit step.

## Background (root causes, verified)

1. **Doc-ID mismatch**: golden facts are `doc_818633`; chunk `doc_id` is `doc_818633_txt` (loader in `src/graphrag/ingest/loader.py:20-22` replaces `.` with `_`). Eval compares `chunk.doc_id == fact` (`src/graphrag/eval/metrics.py:145` and `:212`) → always False.
2. **Missing indexes**: `data/processed/` has only `manifest.jsonl` + a stale `vector_index.pkl`. No `nodes.jsonl`, no `fts_index.sqlite`. The last eval ran against the old 3-chunk graph — every question retrieved the same 3 chunk IDs.
3. **MockLLM hardcodes** Eiffel/Statue-of-Liberty chunk IDs (`src/graphrag/eval/metrics.py:60-70`) → answers "Insufficient evidence" for every HotpotQA question (matches `eval_report.json`).
4. **Duplicate supporting facts** (e.g. `doc_315870` ×2) unfairly halve recall.
5. **Wrong data dir**: corpus lives in `data/raw_full/`, config points at `data/raw/`.

---

### Task 1: Data layout — `data_hotpot/` with symlinks + .gitignore

**Files:**
- Create: `data_hotpot/raw` (symlink → `raw_full`)
- Create: `data_hotpot/eval/golden.jsonl` (symlink → `../data/eval/golden.jsonl`)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: existing `data/raw_full/` corpus (1,977 `.txt` docs), existing `data/eval/golden.jsonl` (20 questions).
- Produces: a data root usable by every CLI command via `GRAPHRAG_DATA_DIR=data_hotpot`.

- [ ] **Step 1: Create the data dir with symlinks**

```bash
mkdir -p data_hotpot/eval
ln -sfn ../data/raw_full data_hotpot/raw
ln -sfn ../../data/eval/golden.jsonl data_hotpot/eval/golden.jsonl
ls -la data_hotpot data_hotpot/eval
```

Expected: `data_hotpot/raw` → `../data/raw_full`, `data_hotpot/eval/golden.jsonl` → `../../data/eval/golden.jsonl`.

- [ ] **Step 2: Verify settings resolve correctly**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli info
```

Expected: `data_dir=data_hotpot`, `raw_dir=data_hotpot/raw` (resolves through the symlink; loader uses `Path.rglob` which follows it).

- [ ] **Step 3: Update .gitignore**

Append to the "Local data" section:

```
data/raw_full/
data/hotpotqa_corpus/
data/hotpotqa_golden.jsonl
data_hotpot/
eval_report.json
```

- [ ] **Step 4: Verify gitignore takes effect**

```bash
git status --short | grep -E "raw_full|hotpot|eval_report|data_hotpot"
```

Expected: no matches (all ignored). The symlinked `data_hotpot` dir contents are ignored as a whole.

---

### Task 2: metrics.py — doc-ID matching, fact dedupe, retrieved_doc_ids

**Files:**
- Modify: `src/graphrag/eval/metrics.py`
- Modify: `tests/test_eval.py` (rewrite in standalone-runner style; it is currently pytest-only and unrunnable)

**Interfaces:**
- Consumes: `GoldenQuestion(id, question, expected_answer, supporting_facts)`, `RetrievedChunk(chunk_id, doc_id, text, score)` from `graphrag.retrieve`.
- Produces (used by Task 3 and later tasks):
  - `_doc_id_matches(chunk_doc_id: str, fact: str) -> bool`
  - `MockLLMClient` class (marker; Task 3 fills behavior)
  - `generate_answer_with_client(client, question, evidence_chunks: list[tuple[str, str]]) -> tuple[str, list[str], float]` (signature unchanged)
  - `EvalResult` gains field `retrieved_doc_ids: list[str]`
  - `run_eval(..., limit: Optional[int] = None)` (Task 4 wires CLI)

- [ ] **Step 1: Write the failing tests (full rewrite of tests/test_eval.py)**

Replace the entire contents of `tests/test_eval.py`:

```python
from pathlib import Path
import tempfile

from graphrag.eval.metrics import (
    MockLLMClient,
    _doc_id_matches,
    generate_answer_with_client,
    load_golden_dataset,
)


def test_load_golden_dataset_dedupes_supporting_facts() -> None:
    with tempfile.TemporaryDirectory() as d:
        eval_dir = Path(d)
        golden_file = eval_dir / "golden.jsonl"
        golden_file.write_text(
            '{"id": "q1", "question": "test?", "expected_answer": "ans",'
            ' "supporting_facts": ["doc1", "doc1", "doc2"]}\n'
        )
        questions = load_golden_dataset(eval_dir)
        assert questions[0].supporting_facts == ["doc1", "doc2"]


def test_doc_id_matches() -> None:
    assert _doc_id_matches("doc_818633_txt", "doc_818633")
    assert _doc_id_matches("doc_818633", "doc_818633")
    assert not _doc_id_matches("doc_999999_txt", "doc_818633")
    # Underscore sentinel prevents prefix collisions between doc IDs
    assert not _doc_id_matches("doc_8186337_txt", "doc_818633")


def test_extractive_mock_answers_and_cites() -> None:
    evidence = [
        ("aaaa", "The Eiffel Tower was built by Gustave Eiffel's company."),
        ("bbbb", "Unrelated text about dolphins."),
    ]
    answer, citations, conf = generate_answer_with_client(
        MockLLMClient(), "Who built the Eiffel Tower?", evidence
    )
    assert citations == ["aaaa"]
    assert "[aaaa]" in answer
    assert conf > 0.5


def test_extractive_mock_no_evidence() -> None:
    answer, citations, conf = generate_answer_with_client(
        MockLLMClient(), "Any question?", []
    )
    assert "Insufficient evidence" in answer
    assert citations == []
    assert conf == 0.1


def test_extractive_mock_citations_always_valid() -> None:
    evidence = [("chunk1", "Some sentence here."), ("chunk2", "Another sentence.")]
    answer, citations, conf = generate_answer_with_client(
        MockLLMClient(), "some sentence?", evidence
    )
    valid_ids = {cid for cid, _ in evidence}
    assert all(c in valid_ids for c in citations)


if __name__ == "__main__":
    import sys

    tests = [
        test_load_golden_dataset_dedupes_supporting_facts,
        test_doc_id_matches,
        test_extractive_mock_answers_and_cites,
        test_extractive_mock_no_evidence,
        test_extractive_mock_citations_always_valid,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            sys.exit(1)
    print("All tests passed!")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src python3 tests/test_eval.py`
Expected: FAIL — `ImportError: cannot import name '_doc_id_matches'` (and the old hardcoded mock fails the extractive tests).

- [ ] **Step 3: Implement in src/graphrag/eval/metrics.py**

3a. Rewrite `load_golden_dataset` to dedupe supporting facts (preserving order). Replace the body of the parse loop (current lines 26-36) so the function reads:

```python
def load_golden_dataset(eval_dir: Path) -> list[GoldenQuestion]:
    path = eval_dir / "golden.jsonl"
    if not path.exists():
        return []
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Handle both 'id' and 'question_id' field names
            if "question_id" in data and "id" not in data:
                data["id"] = data.pop("question_id")
            # HotpotQA repeats fact doc IDs; dedupe for fair recall
            data["supporting_facts"] = list(dict.fromkeys(data.get("supporting_facts", [])))
            questions.append(GoldenQuestion(**data))
    return questions
```

3b. Add the doc-ID matcher and shared retrieval scorer near the top of the module (after imports, before `GoldenQuestion`-adjacent dataclasses is fine):

```python
def _doc_id_matches(chunk_doc_id: str, fact: str) -> bool:
    """Match a golden fact doc ID (e.g. 'doc_818633') against a chunk doc_id
    (e.g. 'doc_818633_txt', produced by the loader's '.'->'_' rename)."""
    if chunk_doc_id == fact:
        return True
    return chunk_doc_id.startswith(fact + "_")


def _score_retrieval(
    gq: GoldenQuestion, retrieved_chunks: list
) -> tuple[bool, float]:
    """Compute (hit_at_k, recall_at_k) for one golden question."""
    hit = False
    matched = 0
    for fact in gq.supporting_facts:
        for chunk in retrieved_chunks:
            if _doc_id_matches(chunk.doc_id, fact):
                hit = True
                matched += 1
                break
    recall = matched / len(gq.supporting_facts) if gq.supporting_facts else 0.0
    return hit, recall
```

3c. Add `retrieved_doc_ids: list[str]` to `EvalResult` (after `retrieved_chunk_ids`):

```python
@dataclass(slots=True)
class EvalResult:
    question_id: str
    question: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    expected_supporting_facts: list[str]
    hit_at_k: bool
    recall_at_k: float
    citations_valid: bool
    answer: str
    confidence: float
```

3d. In `evaluate_retrieval` (current lines 126-165), replace the duplicated hit/recall loop with the helper and populate the new field:

```python
    results = []
    for gq in golden:
        retrieval = retrieve_context(
            store, processed_dir, gq.question, top_k_chunks=top_k, graph_hops=hops
        )
        retrieved_chunks = retrieval["all_chunks"]
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_doc_ids = sorted({c.doc_id for c in retrieved_chunks})

        hit, recall = _score_retrieval(gq, retrieved_chunks)

        results.append(
            EvalResult(
                question_id=gq.id,
                question=gq.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_supporting_facts=gq.supporting_facts,
                hit_at_k=hit,
                recall_at_k=recall,
                citations_valid=False,
                answer="",
                confidence=0.0,
            )
        )
    return results
```

- [ ] **Step 4: Run tests to verify Task 2 portion passes**

Run: `PYTHONPATH=src python3 tests/test_eval.py`
Expected: `test_load_golden_dataset_dedupes_supporting_facts` and `test_doc_id_matches` PASS; the three mock tests still FAIL (mock not yet rewritten — Task 3).

- [ ] **Step 5: No commit yet** (commit in Task 4 once metrics.py changes are complete and green).

---

### Task 3: metrics.py — extractive MockLLMClient + simplified mock path

**Files:**
- Modify: `src/graphrag/eval/metrics.py:53-123` (MockLLMClient + generate_answer_with_client), `:207-216` (full-pipeline hit/recall)

**Interfaces:**
- Consumes: `_score_retrieval`, `_doc_id_matches`, `EvalResult.retrieved_doc_ids` from Task 2.
- Produces: `MockLLMClient` (marker class), `_extractive_answer(question, evidence_chunks) -> tuple[str, list[str], float]`. `generate_answer_with_client` signature unchanged, now routes mock clients to `_extractive_answer`.

- [ ] **Step 1: Rewrite MockLLMClient and generate_answer_with_client**

Replace `MockLLMClient` (current lines 53-70) and `generate_answer_with_client` (current lines 73-123) with:

```python
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "was", "were", "is",
    "are", "and", "or", "what", "who", "which", "did", "do", "does", "by",
    "for", "with", "same", "their", "his", "her", "its", "that", "this",
}


class MockLLMClient:
    """Deterministic extractive mock. generate_answer_with_client handles it
    specially — no HTTP client, no hardcoded corpus knowledge."""


def _extractive_answer(
    question: str, evidence_chunks: list[tuple[str, str]]
) -> tuple[str, list[str], float]:
    """Pick the evidence chunk with the highest question-token overlap and
    answer with its first sentence, citing that chunk."""
    if not evidence_chunks:
        return "Insufficient evidence to answer.", [], 0.1

    q_tokens = set(re.findall(r"\w+", question.lower())) - _STOPWORDS
    best_id, best_text, best_score = evidence_chunks[0][0], evidence_chunks[0][1], -1.0
    for cid, text in evidence_chunks:
        t_tokens = set(re.findall(r"\w+", text.lower()))
        score = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        if score > best_score:
            best_id, best_text, best_score = cid, text, score

    sentences = re.split(r"(?<=[.!?])\s+", best_text.strip())
    sentence = (sentences[0] if sentences else best_text)[:300]
    return f"{sentence} [{best_id}]", [best_id], 0.8


def generate_answer_with_client(
    client, question: str, evidence_chunks: list[tuple[str, str]]
) -> tuple[str, list[str], float]:
    """Generate answer and extract citations. Mock clients are answered
    extractively; real clients go through graphrag.generate."""
    if isinstance(client, MockLLMClient):
        return _extractive_answer(question, evidence_chunks)

    from graphrag.generate import AnswerResult, generate_answer

    result: AnswerResult = generate_answer(client, question, evidence_chunks)
    return result.answer, result.citations, result.confidence
```

Note: the top-level `import re` already exists in metrics.py (line 4).

- [ ] **Step 2: Use the shared scorer in evaluate_full_pipeline**

In `evaluate_full_pipeline` (current lines 207-216), replace the duplicated hit/recall block:

```python
        retrieved_chunks = retrieval["all_chunks"]
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_doc_ids = sorted({c.doc_id for c in retrieved_chunks})

        # Check retrieval hit
        hit, recall = _score_retrieval(gq, retrieved_chunks)
```

and pass `retrieved_doc_ids=retrieved_doc_ids` in the `EvalResult(...)` construction (current lines 230-242).

- [ ] **Step 3: Run all tests to verify green**

Run: `PYTHONPATH=src python3 tests/test_eval.py`
Expected: All 5 tests PASS.

- [ ] **Step 4: Run the other test suites (no regressions)**

Run: `PYTHONPATH=src python3 tests/test_extract.py && PYTHONPATH=src python3 tests/test_cfg.py`
Expected: "All tests passed!" for both. (`tests/test_cfg.py` has no `__main__` runner — if it fails to execute, verify it contains only assertions at module level and still passes when imported; it did pass `python3 tests/test_cfg.py` previously as a bare import.)

- [ ] **Step 5: No commit yet** (commit in Task 4).

---

### Task 4: CLI `--limit` flag + wire through run_eval

**Files:**
- Modify: `src/graphrag/eval/metrics.py` (`run_eval`, current lines 246-258)
- Modify: `src/graphrag/cli.py:186-211` (cmd_eval) and eval parser setup (near line 485)

**Interfaces:**
- Consumes: `run_eval(settings, top_k, hops, output_path, use_mock_llm, full_pipeline)` — current signature.
- Produces: `run_eval(..., limit: Optional[int] = None)`; CLI flag `--limit N`.

- [ ] **Step 1: Add limit to run_eval**

Change the signature and golden handling:

```python
def run_eval(
    settings: Settings,
    top_k: int = 5,
    hops: int = 1,
    output_path: Optional[Path] = None,
    use_mock_llm: bool = False,
    full_pipeline: bool = False,
    limit: Optional[int] = None,
) -> dict:
    golden = load_golden_dataset(settings.eval_dir)
    if not golden:
        raise RuntimeError(
            f"No golden questions found in {settings.eval_dir}/golden.jsonl"
        )
    if limit:
        golden = golden[:limit]
```

(rest of the function unchanged)

- [ ] **Step 2: Wire the CLI flag**

In `cmd_eval` (cli.py:190-197), add to the `run_eval(...)` call:

```python
            limit=args.limit,
```

In the eval parser setup, add before `eval_parser.set_defaults(func=cmd_eval)`:

```python
    eval_parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N questions"
    )
```

- [ ] **Step 3: Verify flag works without a full eval**

Run: `GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli eval --top-k 3 --limit 2`
Expected: Either an eval report over 2 questions, or `Eval failed: Graph is empty...` — both acceptable at this point; the flag must not crash argparse. (Real numbers come in Task 5/6.)

- [ ] **Step 4: Commit all metrics/CLI/test work**

```bash
git add src/graphrag/eval/metrics.py src/graphrag/cli.py tests/test_eval.py .gitignore data/eval/golden.jsonl src/graphrag/config.py src/graphrag/extract/pipeline.py src/graphrag/llm/client.py
git rm --cached data/raw/.gitkeep 2>/dev/null || true
git commit -m "feat(eval): doc-level HotpotQA matching, extractive mock LLM, --limit flag"
```

NOTE for executor: `config.py`, `extract/pipeline.py`, `llm/client.py`, `golden.jsonl`, and the `.gitkeep` deletion are pre-existing uncommitted work bundled with this effort — confirm with the user before staging them.

---

### Task 5: Rebuild pipeline state for the HotpotQA corpus

**Files:** none (state rebuild — writes `data_hotpot/processed/*`, all gitignored)

**Interfaces:**
- Consumes: `data_hotpot/` layout from Task 1; structure-only `build-graph` path (cli.py:268-275 saves Doc/Chunk/HAS_CHUNK graph when no API key is set).
- Produces: `data_hotpot/processed/{manifest.jsonl, nodes.jsonl, edges.jsonl, fts_index.sqlite, vector_index.pkl}` — the state `eval` requires.

- [ ] **Step 1: Ingest**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli ingest
```

Expected: `Ingested 1977 docs -> 2142 chunks (manifest written)` (counts may vary by a few; should be ~2,142).

- [ ] **Step 2: Build structure-only graph (no API key needed)**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli build-graph
```

Expected output includes: `LLM API key not set (GRAPHRAG_LLM_API_KEY). Extraction skipped.` and `Graph structure (documents/chunks) saved.` Verify: `wc -l data_hotpot/processed/nodes.jsonl` → ~4,120 (1,977 doc + 2,142 chunk nodes).

- [ ] **Step 3: Build FTS index**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli build-fts
```

Expected: `FTS5 index built successfully.` and `data_hotpot/processed/fts_index.sqlite` exists.

- [ ] **Step 4: Build vector index (overwrites stale pkl)**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli build-vector
```

Expected: `Vector index built successfully.` (CPU fastembed over 2,142 chunks — allow a few minutes). If it fails with `fastembed not installed`, run `pip install fastembed` and retry.

---

### Task 6: Run evals, record baseline, update README

**Files:**
- Modify: `README.md` (Evaluation Results section, lines 112-122)
- Create (gitignored): `eval_report.json` regenerated by `--output`

**Interfaces:**
- Consumes: all prior tasks; golden dataset at `data_hotpot/eval/golden.jsonl`.

- [ ] **Step 1: Retrieval-only eval (the honest baseline)**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli eval --top-k 5 --output eval_report.json
```

Expected: non-zero hit rate and recall (retrieval now matches against the right corpus with fixed doc-ID logic). Record `hit_rate`, `avg_recall_at_k`.

- [ ] **Step 2: Full-pipeline smoke test with mock (limit 5)**

```bash
GRAPHRAG_DATA_DIR=data_hotpot PYTHONPATH=src python3 -m graphrag.cli eval --top-k 5 --mock-llm --full-pipeline --limit 5 --output eval_report_mock.json
```

Expected: `citation_valid_rate` 100% (extractive mock always cites retrieved chunks). Answers are extractive sentences, not "Insufficient evidence" (unless evidence was empty).

- [ ] **Step 3: Update README evaluation section**

Replace the table in README.md (lines 114-121) with measured values, e.g.:

```markdown
| Metric | Retrieval-only (HotpotQA, n=20) |
|--------|----------------------------------|
| **Hit Rate@5** | <measured> |
| **Avg Recall@5** | <measured> |
| **Citation Valid Rate** | 100% (extractive mock, n=5 subset) |

> Evaluated on a 1,977-doc HotpotQA corpus (2,142 chunks), 20 multi-hop questions.
> Retrieval is FTS5 + vector only — no entity graph (extraction not run at this scale), so multi-hop questions score low. See Limitations.
```

Keep the existing caveat paragraph below the table, updated to the new corpus size.

- [ ] **Step 4: Sanity-check the report details**

```bash
python3 -c "
import json
r = json.load(open('eval_report.json'))
print('hit_rate:', r['hit_rate'], 'recall:', r['avg_recall_at_k'])
print('sample doc_ids:', r['details'][0]['retrieved_doc_ids'])
"
```

Expected: `retrieved_doc_ids` are `doc_*` strings (proves doc-level matching is live).

---

### Task 7: Final verification + commit remainder

- [ ] **Step 1: Full test suite**

```bash
PYTHONPATH=src python3 tests/test_eval.py && PYTHONPATH=src python3 tests/test_extract.py
```

Expected: "All tests passed!" twice.

- [ ] **Step 2: Commit README update**

```bash
git add README.md
git commit -m "docs: HotpotQA baseline eval results (1,977-doc corpus)"
```

- [ ] **Step 3: Confirm clean tree**

```bash
git status --short
```

Expected: nothing untracked or modified outside `data_hotpot/`, `data/raw_full/`, `data/hotpotqa*`, `eval_report*.json`, `data/processed/` (all gitignored).

## Self-Review Notes

- Spec coverage: doc-ID mismatch (Task 2), dedupe (Task 2), stale indexes (Task 5), hardcoded mock (Task 3), raw-dir location (Task 1), cost control via --limit (Task 4), verification (Tasks 5-6). ✅
- Type consistency: `EvalResult.retrieved_doc_ids` added in Task 2 and populated in Tasks 2/3; `_score_retrieval` defined Task 2, consumed Task 3; `run_eval(limit=...)` defined Task 4, consumed by CLI in same task. ✅
- `evaluate_retrieval`'s `retrieved_chunks` are `RetrievedChunk` objects — `_score_retrieval` accesses `.doc_id`; both call sites pass `RetrievedChunk` lists. ✅
