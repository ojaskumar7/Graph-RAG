from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from graphrag.llm.client import OpenAICompatibleClient, LLMError


ANSWER_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided evidence.

Evidence chunks (each has an ID):
{evidence}

Question: {question}

Instructions:
- Answer ONLY using the evidence above
- Cite evidence by chunk ID in square brackets, e.g., [abc123]
- If evidence is insufficient, say "Insufficient evidence to answer."
- Be concise and direct

Answer:"""


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[str]  # chunk IDs
    evidence_used: list[str]  # chunk IDs
    confidence: float


def generate_answer(
    client: OpenAICompatibleClient,
    question: str,
    evidence_chunks: list[tuple[str, str]],  # (chunk_id, text)
    max_evidence_chars: int = 8000,
) -> AnswerResult:
    # Format evidence
    evidence_parts = []
    total_chars = 0
    for chunk_id, text in evidence_chunks:
        if total_chars + len(text) > max_evidence_chars:
            break
        evidence_parts.append(f"[Chunk {chunk_id}]\n{text}")
        total_chars += len(text)

    evidence_str = "\n\n---\n\n".join(evidence_parts)

    messages = [
        {
            "role": "user",
            "content": ANSWER_PROMPT.format(evidence=evidence_str, question=question),
        }
    ]

    try:
        response = client.chat(messages, temperature=0.0)
    except LLMError as e:
        return AnswerResult(
            answer=f"Error generating answer: {e}",
            citations=[],
            evidence_used=[cid for cid, _ in evidence_chunks],
            confidence=0.0,
        )

    # Extract citations from response
    import re

    citations = list(set(re.findall(r"\[([a-f0-9]{16})\]", response)))
    # Also match [Chunk abc123] format
    chunk_citations = list(set(re.findall(r"\[Chunk ([a-f0-9]{16})\]", response)))
    all_citations = list(set(citations + chunk_citations))

    # Validate citations against evidence
    evidence_ids = {cid for cid, _ in evidence_chunks}
    valid_citations = [c for c in all_citations if c in evidence_ids]

    # Simple confidence heuristic
    has_insufficient = "insufficient evidence" in response.lower()
    confidence = 0.1 if has_insufficient else (0.8 if valid_citations else 0.4)

    return AnswerResult(
        answer=response.strip(),
        citations=valid_citations,
        evidence_used=list(evidence_ids),
        confidence=confidence,
    )
