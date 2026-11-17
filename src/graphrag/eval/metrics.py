from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from graphrag.config import Settings
from graphrag.graph import GraphStore, NodeType
from graphrag.retrieve import retrieve_context


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


@dataclass(slots=True)
class GoldenQuestion:
    id: str
    question: str
    expected_answer: str
    supporting_facts: list[str]  # doc/chunk hints


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
            data["supporting_facts"] = list(
                dict.fromkeys(data.get("supporting_facts", []))
            )
            questions.append(GoldenQuestion(**data))
    return questions


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


def evaluate_retrieval(
    store: GraphStore,
    processed_dir: Path,
    golden: list[GoldenQuestion],
    top_k: int = 5,
    hops: int = 1,
) -> list[EvalResult]:
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


def evaluate_full_pipeline(
    settings: Settings,
    golden: list[GoldenQuestion],
    top_k: int = 5,
    hops: int = 1,
    use_mock_llm: bool = False,
) -> list[EvalResult]:
    """Run full pipeline eval with citation validation."""
    from graphrag.llm.client import OpenAICompatibleClient
    from graphrag.generate import generate_answer

    store = GraphStore(settings.processed_dir)
    if not store.nodes:
        raise RuntimeError("Graph is empty. Run 'build-graph' first.")

    # Use mock or real LLM
    if use_mock_llm or not settings.llm_api_key:
        client = MockLLMClient()
    else:
        client = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

    results = []
    for gq in golden:
        retrieval = retrieve_context(
            store,
            settings.processed_dir,
            gq.question,
            top_k_chunks=top_k,
            graph_hops=hops,
        )
        retrieved_chunks = retrieval["all_chunks"]
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_doc_ids = sorted({c.doc_id for c in retrieved_chunks})

        # Check retrieval hit
        hit, recall = _score_retrieval(gq, retrieved_chunks)

        # Generate answer with citations
        evidence = [(c.chunk_id, c.text) for c in retrieved_chunks]
        answer, citations, confidence = generate_answer_with_client(
            client, gq.question, evidence
        )

        # Validate citations: all cited chunks must be in retrieved set
        evidence_ids = set(retrieved_chunk_ids)
        citations_valid = (
            all(c in evidence_ids for c in citations) and len(citations) > 0
        )

        results.append(
            EvalResult(
                question_id=gq.id,
                question=gq.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_supporting_facts=gq.supporting_facts,
                hit_at_k=hit,
                recall_at_k=recall,
                citations_valid=citations_valid,
                answer=answer,
                confidence=confidence,
            )
        )
    return results


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

    if full_pipeline:
        results = evaluate_full_pipeline(settings, golden, top_k, hops, use_mock_llm)
    else:
        store = GraphStore(settings.processed_dir)
        results = evaluate_retrieval(store, settings.processed_dir, golden, top_k, hops)

    total = len(results)
    hits = sum(1 for r in results if r.hit_at_k)
    avg_recall = sum(r.recall_at_k for r in results) / total if total > 0 else 0.0
    citation_valid_count = sum(1 for r in results if r.citations_valid)
    citation_valid_rate = citation_valid_count / total if total > 0 else 0.0

    report = {
        "total_questions": total,
        "hit_at_k": hits,
        "hit_rate": hits / total if total > 0 else 0.0,
        "avg_recall_at_k": avg_recall,
        "citations_valid_count": citation_valid_count,
        "citation_valid_rate": citation_valid_rate,
        "details": [asdict(r) for r in results],
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
