from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from graphrag.llm.client import OpenAICompatibleClient, LLMError


EXTRACTION_PROMPT = """You are an entity and relation extractor. Given a text chunk, extract entities and relations in JSON format.

Output ONLY valid JSON with this schema:
{
  "entities": [
    {"name": "string", "type": "string", "canonical_name": "string"}
  ],
  "relations": [
    {"subject": "string", "predicate": "string", "object": "string", "confidence": 0.0-1.0}
  ]
}

Rules:
- Entities: people, places, organizations, events, concepts, works
- Types: Person, Location, Organization, Event, Work, Concept, etc.
- Relations: use clear predicates (e.g., "designed", "built", "located_in", "gifted_to", "capital_of")
- canonical_name: normalized form (e.g., "Gustave Eiffel" not "Eiffel")
- Only extract what is explicitly stated in the text
- confidence: 0.0 to 1.0 based on clarity in text

Text:
{text}"""


@dataclass(slots=True)
class ExtractedEntity:
    name: str
    entity_type: str
    canonical_name: str


@dataclass(slots=True)
class ExtractedRelation:
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass(slots=True)
class ExtractionResult:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


class ExtractionPipeline:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        self.client = client
        self.max_retries = max_retries
        self.temperature = temperature

    def extract(self, chunk_text: str) -> ExtractionResult:
        for attempt in range(self.max_retries + 1):
            try:
                result = self._extract_once(chunk_text)
                return result
            except (json.JSONDecodeError, LLMError) as e:
                if attempt >= self.max_retries:
                    raise ExtractionError(
                        f"Extraction failed after {self.max_retries + 1} attempts: {e}"
                    ) from e
                # Try repair on next attempt
                continue
        raise ExtractionError("Extraction failed unexpectedly")

    def _extract_once(self, chunk_text: str) -> ExtractionResult:
        prompt = EXTRACTION_PROMPT.format(text=chunk_text)
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(messages, temperature=self.temperature)

        # Try to parse JSON directly
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            data = self._repair_json(response)

        # Validate and convert
        entities = []
        for ent in data.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=ent.get("type", "Concept").strip(),
                    canonical_name=ent.get("canonical_name", name).strip(),
                )
            )

        relations = []
        for rel in data.get("relations", []):
            subj = rel.get("subject", "").strip()
            pred = rel.get("predicate", "").strip()
            obj = rel.get("object", "").strip()
            if not (subj and pred and obj):
                continue
            try:
                conf = float(rel.get("confidence", 0.5))
            except (ValueError, TypeError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            relations.append(
                ExtractedRelation(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    confidence=conf,
                )
            )

        return ExtractionResult(entities=entities, relations=relations)

    def _repair_json(self, response: str) -> dict:
        """Attempt to extract valid JSON from LLM response."""
        # Try to find JSON object in response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        # If no JSON found, return empty structure
        return {"entities": [], "relations": []}


class ExtractionError(Exception):
    pass
