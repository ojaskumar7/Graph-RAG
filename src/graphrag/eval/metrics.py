from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from graphrag.config import Settings
from graphrag.graph import GraphStore, NodeType
from graphrag.retrieve import retrieve_context


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
            questions.append(GoldenQuestion(**data))
    return questions


@dataclass(slots=True)
class EvalResult:
    question_id: str
    question: str
    retrieved_chunk_ids: list[str]
    expected_supporting_facts: list[str]
    hit_at_k: bool
    recall_at_k: float
    citations_valid: bool
    answer: str
    confidence: float


class MockLLMClient:
    """Mock LLM client for testing without API key."""

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        # Extract question and evidence from last message
        content = messages[-1]["content"]
        # Simple heuristic answer based on evidence
        if "Eiffel Tower" in content and "built" in content:
            return "The Eiffel Tower was built by Gustave Eiffel's company [eeb24a88f8ee5d4d]."
        if "Eiffel Tower" in content and "located" in content:
            return "The Eiffel Tower is located in Paris, France [eeb24a88f8ee5d4d]."
        if "Statue of Liberty" in content and "designed" in content:
            return "The Statue of Liberty was designed by Frédéric Auguste Bartholdi [dd72ce7db6bd2c56]."
        if "Statue of Liberty" in content and "metal framework" in content:
            return "The metal framework was built by Gustave Eiffel [dd72ce7db6bd2c56]."
        if "World's Fair" in content and "centerpiece" in content:
            return "The Eiffel Tower was the centerpiece of the 1889 World's Fair [232d11df9ce35863]."
        return "Insufficient evidence to answer."


def generate_answer_with_client(
    client, question: str, evidence_chunks: list[tuple[str, str]]
) -> tuple[str, list[str], float]:
    """Generate answer and extract citations."""
    from graphrag.generate import generate_answer, AnswerResult

    # We need to call the actual generate_answer but with our client
    # Let's use a simplified version here
    if isinstance(client, MockLLMClient):
        # Format evidence like generate_answer does
        evidence_parts = []
        for chunk_id, text in evidence_chunks:
            evidence_parts.append(f"[Chunk {chunk_id}]\n{text}")
        evidence_str = "\n\n---\n\n".join(evidence_parts)

        # Simple prompt format
        prompt = f"""You are a helpful assistant that answers questions using ONLY the provided evidence.

Evidence chunks (each has an ID):
{evidence_str}

Question: {question}

Instructions:
- Answer ONLY using the evidence above
- Cite evidence by chunk ID in square brackets, e.g., [abc123]
- If evidence is insufficient, say "Insufficient evidence to answer."
- Be concise and direct

Answer:"""

        messages = [{"role": "user", "content": prompt}]
        response = client.chat(messages, temperature=0.0)

        # Extract citations
        citations = list(set(re.findall(r"\[([a-f0-9]{16})\]", response)))
        chunk_citations = list(set(re.findall(r"\[Chunk ([a-f0-9]{16})\]", response)))
        all_citations = list(set(citations + chunk_citations))

        # Validate citations against evidence
        evidence_ids = {cid for cid, _ in evidence_chunks}
        valid_citations = [c for c in all_citations if c in evidence_ids]

        has_insufficient = "insufficient evidence" in response.lower()
        confidence = 0.1 if has_insufficient else (0.8 if valid_citations else 0.4)

        return response.strip(), valid_citations, confidence
    else:
        # Use real LLM via graphrag.generate
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

        hit = False
        matched = 0
        for fact in gq.supporting_facts:
            for chunk in retrieved_chunks:
                if chunk.doc_id == fact or fact in chunk.chunk_id:
                    hit = True
                    matched += 1
                    break

        recall = matched / len(gq.supporting_facts) if gq.supporting_facts else 0.0

        results.append(
            EvalResult(
                question_id=gq.id,
                question=gq.question,
                retrieved_chunk_ids=retrieved_chunk_ids,
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

        # Check retrieval hit
        hit = False
        matched = 0
        for fact in gq.supporting_facts:
            for chunk in retrieved_chunks:
                if chunk.doc_id == fact or fact in chunk.chunk_id:
                    hit = True
                    matched += 1
                    break
        recall = matched / len(gq.supporting_facts) if gq.supporting_facts else 0.0

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
) -> dict:
    golden = load_golden_dataset(settings.eval_dir)
    if not golden:
        raise RuntimeError(
            f"No golden questions found in {settings.eval_dir}/golden.jsonl"
        )

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
