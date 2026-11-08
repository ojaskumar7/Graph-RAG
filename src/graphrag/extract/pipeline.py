from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from graphrag.llm.client import OpenAICompatibleClient, LLMError


EXTRACTION_PROMPT = """You are an entity and relation extractor. Given a text chunk, extract entities and relations in JSON format.

Output ONLY valid JSON with this schema:
{{
  "entities": [
    {{"name": "string", "type": "string", "canonical_name": "string"}}
  ],
  "relations": [
    {{"subject": "string", "predicate": "string", "object": "string", "confidence": 0.0-1.0}}
  ]
}}

Rules:
- Entities: people, places, organizations, events, concepts, works
- Types: Person, Location, Organization, Event, Work, Concept, etc.
- Relations: use clear predicates (e.g., "designed", "built", "located_in", "gifted_to", "capital_of")
- canonical_name: normalized form (e.g., "Gustave Eiffel" not "Eiffel")
- Only extract what is explicitly stated in the text
- confidence: 0.0 to 1.0 based on clarity in text

Text:
{text}"""

RETRY_PROMPT = """Your previous response was not valid JSON. Please output ONLY valid JSON matching this exact schema:

{{
  "entities": [
    {{"name": "string", "type": "string", "canonical_name": "string"}}
  ],
  "relations": [
    {{"subject": "string", "predicate": "string", "object": "string", "confidence": 0.0-1.0}}
  ]
}}

Common mistakes to avoid:
- No trailing commas
- All strings must be in double quotes
- confidence must be a number between 0.0 and 1.0
- No comments or extra text outside the JSON object

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


@dataclass(slots=True)
class ExtractionMetrics:
    attempts: int
    success: bool
    parse_error: Optional[str] = None
    validation_errors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


class ExtractionPipeline:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        max_retries: int = 2,
        temperature: float = 0.0,
        strict_mode: bool = True,
    ):
        self.client = client
        self.max_retries = max_retries
        self.temperature = temperature
        self.strict_mode = strict_mode
        self._metrics: list[ExtractionMetrics] = []

    def extract(self, chunk_text: str) -> ExtractionResult:
        self._metrics.clear()

        for attempt in range(self.max_retries + 1):
            try:
                result, metrics = self._extract_once(chunk_text, attempt)
                self._metrics.append(metrics)
                return result
            except (json.JSONDecodeError, LLMError) as e:
                # Retry on parse/LLM errors
                metrics = ExtractionMetrics(
                    attempts=attempt + 1,
                    success=False,
                    parse_error=str(e),
                )
                self._metrics.append(metrics)
                if attempt >= self.max_retries:
                    raise ExtractionError(
                        f"Extraction failed after {self.max_retries + 1} attempts: {e}"
                    ) from e
                continue
            except ValidationError:
                # In strict mode, don't retry validation errors - raise immediately
                if self.strict_mode:
                    raise
                # In non-strict mode, record and continue
                metrics = ExtractionMetrics(
                    attempts=attempt + 1,
                    success=False,
                    parse_error="Validation error",
                )
                self._metrics.append(metrics)
                if attempt >= self.max_retries:
                    raise ExtractionError(
                        f"Extraction failed after {self.max_retries + 1} attempts: validation errors"
                    )
                continue
        raise ExtractionError("Extraction failed unexpectedly")

    def _extract_once(
        self, chunk_text: str, attempt: int
    ) -> tuple[ExtractionResult, ExtractionMetrics]:
        prompt = (
            RETRY_PROMPT.format(text=chunk_text)
            if attempt > 0
            else EXTRACTION_PROMPT.format(text=chunk_text)
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(messages, temperature=self.temperature)

        # Try to parse JSON directly
        try:
            data = json.loads(response)
            parse_error = None
        except json.JSONDecodeError as e:
            # Try to extract/repair JSON from response
            data = self._repair_json(response)
            if data is None:
                # Re-raise original JSONDecodeError to trigger retry
                raise
            parse_error = f"JSON decode error: {e}"

        # Validate structure
        validation_errors = self._validate_structure(data)
        if validation_errors and self.strict_mode:
            raise ValidationError(f"Validation errors: {'; '.join(validation_errors)}")

        # Convert to typed objects
        entities = []
        for i, ent in enumerate(data.get("entities", [])):
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
        for i, rel in enumerate(data.get("relations", [])):
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

        metrics = ExtractionMetrics(
            attempts=attempt + 1,
            success=True,
            parse_error=parse_error,
            validation_errors=validation_errors,
        )
        return ExtractionResult(entities=entities, relations=relations), metrics

    def _validate_structure(self, data: dict) -> list[str]:
        """Validate JSON structure matches expected schema."""
        errors = []
        if not isinstance(data, dict):
            errors.append("Root must be an object")
            return errors

        # Check entities
        entities = data.get("entities")
        if entities is None:
            errors.append("Missing 'entities' field")
        elif not isinstance(entities, list):
            errors.append("'entities' must be a list")
        else:
            for i, ent in enumerate(entities):
                if not isinstance(ent, dict):
                    errors.append(f"Entity {i}: must be an object")
                    continue
                if "name" not in ent or not ent.get("name", "").strip():
                    errors.append(f"Entity {i}: missing or empty 'name'")

        # Check relations
        relations = data.get("relations")
        if relations is None:
            errors.append("Missing 'relations' field")
        elif not isinstance(relations, list):
            errors.append("'relations' must be a list")
        else:
            for i, rel in enumerate(relations):
                if not isinstance(rel, dict):
                    errors.append(f"Relation {i}: must be an object")
                    continue
                missing = [
                    k
                    for k in ("subject", "predicate", "object")
                    if k not in rel or not rel.get(k, "").strip()
                ]
                if missing:
                    errors.append(f"Relation {i}: missing required fields: {missing}")
                if "confidence" in rel:
                    try:
                        float(rel["confidence"])
                    except (ValueError, TypeError):
                        errors.append(f"Relation {i}: confidence must be a number")

        return errors

    def _repair_json(self, response: str) -> Optional[dict]:
        """Attempt to extract and repair valid JSON from LLM response."""
        # Strategy 1: Find JSON in code blocks
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 2: Find first complete JSON object
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group()
            # Try direct parse
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            # Try repair
            repaired = self._repair_common_json_errors(json_str)
            if repaired:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: Try to extract key-value pairs and rebuild
        return self._extract_from_text(response)

    def _repair_common_json_errors(self, json_str: str) -> Optional[str]:
        """Attempt to fix common JSON syntax errors."""
        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # Fix unquoted keys (simple case)
        json_str = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', json_str)
        # Fix single quotes to double quotes (careful - don't break strings)
        # This is risky, so only do it if we can identify it's likely valid
        return json_str

    def _extract_from_text(self, response: str) -> Optional[dict]:
        """Last resort: extract entities/relations from text using regex."""
        entities = []
        relations = []

        # Look for entity-like patterns
        entity_matches = re.findall(r'"name"\s*:\s*"([^"]+)"', response)
        type_matches = re.findall(r'"type"\s*:\s*"([^"]+)"', response)
        canonical_matches = re.findall(r'"canonical_name"\s*:\s*"([^"]+)"', response)

        for i, name in enumerate(entity_matches):
            entities.append(
                {
                    "name": name,
                    "type": type_matches[i] if i < len(type_matches) else "Concept",
                    "canonical_name": canonical_matches[i]
                    if i < len(canonical_matches)
                    else name,
                }
            )

        # Look for relation-like patterns
        subj_matches = re.findall(r'"subject"\s*:\s*"([^"]+)"', response)
        pred_matches = re.findall(r'"predicate"\s*:\s*"([^"]+)"', response)
        obj_matches = re.findall(r'"object"\s*:\s*"([^"]+)"', response)
        conf_matches = re.findall(r'"confidence"\s*:\s*([\d.]+)', response)

        for i, subj in enumerate(subj_matches):
            if i < len(pred_matches) and i < len(obj_matches):
                relations.append(
                    {
                        "subject": subj,
                        "predicate": pred_matches[i],
                        "object": obj_matches[i],
                        "confidence": float(conf_matches[i])
                        if i < len(conf_matches)
                        else 0.5,
                    }
                )

        if entities or relations:
            return {"entities": entities, "relations": relations}
        return None

    def get_metrics(self) -> list[ExtractionMetrics]:
        """Return metrics from all extraction attempts."""
        return self._metrics


class ExtractionError(Exception):
    pass


class ValidationError(Exception):
    pass
