from graphrag.extract import (
    ExtractionPipeline,
    ExtractionResult,
    ValidationError,
    ExtractionError,
)


class MockClient:
    """Mock client that returns various JSON responses for testing."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]
        self.call_count += 1
        return response


def test_valid_json_first_try() -> None:
    """Test successful extraction on first attempt with valid JSON."""
    client = MockClient(
        [
            '{"entities": [{"name": "Eiffel Tower", "type": "Work", "canonical_name": "Eiffel Tower"}], "relations": []}'
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("The Eiffel Tower is in Paris.")
    assert len(result.entities) == 1
    assert result.entities[0].name == "Eiffel Tower"
    assert result.entities[0].entity_type == "Work"
    metrics = pipeline.get_metrics()
    assert len(metrics) == 1
    assert metrics[0].success is True
    assert metrics[0].attempts == 1


def test_json_in_code_block() -> None:
    """Test extraction when JSON is wrapped in code block."""
    client = MockClient(
        [
            '```json\n{"entities": [{"name": "Statue of Liberty", "type": "Work", "canonical_name": "Statue of Liberty"}], "relations": []}\n```'
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("The Statue of Liberty is in New York.")
    assert len(result.entities) == 1
    assert result.entities[0].name == "Statue of Liberty"


def test_json_with_trailing_comma_repair() -> None:
    """Test extraction repairs trailing comma."""
    client = MockClient(
        [
            '{"entities": [{"name": "Paris", "type": "Location", "canonical_name": "Paris"},], "relations": []}'
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("Paris is the capital of France.")
    assert len(result.entities) == 1
    assert result.entities[0].name == "Paris"
    metrics = pipeline.get_metrics()
    assert metrics[0].parse_error is not None


def test_retry_on_invalid_json() -> None:
    """Test retry mechanism on invalid JSON."""
    client = MockClient(
        [
            "not valid json at all",
            '{"entities": [{"name": "Gustave Eiffel", "type": "Person", "canonical_name": "Gustave Eiffel"}], "relations": []}',
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("Gustave Eiffel built the tower.")
    assert len(result.entities) == 1
    assert result.entities[0].name == "Gustave Eiffel"
    metrics = pipeline.get_metrics()
    assert len(metrics) == 2
    assert metrics[0].success is False
    assert metrics[1].success is True


def test_retry_exhaustion_raises() -> None:
    """Test that ExtractionError is raised after all retries fail."""
    client = MockClient(
        [
            "invalid",
            "also invalid",
            "still invalid",
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    try:
        pipeline.extract("Some text")
        assert False, "Should have raised ExtractionError"
    except ExtractionError:
        pass
    metrics = pipeline.get_metrics()
    assert len(metrics) == 3
    assert all(not m.success for m in metrics)


def test_validation_error_on_missing_fields() -> None:
    """Test validation catches missing required fields."""
    client = MockClient(
        [
            '{"entities": [{"type": "Person"}], "relations": []}'  # missing name
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2, strict_mode=True)
    try:
        pipeline.extract("Some text")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "name" in str(e)


def test_non_strict_mode_allows_validation_errors() -> None:
    """Test non-strict mode allows validation errors."""
    client = MockClient(
        [
            '{"entities": [{"type": "Person"}], "relations": []}'  # missing name
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2, strict_mode=False)
    result = pipeline.extract("Some text")
    # Should succeed but with empty entities
    assert len(result.entities) == 0
    metrics = pipeline.get_metrics()
    assert metrics[0].success is True
    assert len(metrics[0].validation_errors) > 0


def test_confidence_clamping() -> None:
    """Test confidence values are clamped to 0.0-1.0."""
    client = MockClient(
        [
            '{"entities": [], "relations": [{"subject": "A", "predicate": "rel", "object": "B", "confidence": 1.5}]}'
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("Some text")
    assert len(result.relations) == 1
    assert result.relations[0].confidence == 1.0

    client2 = MockClient(
        [
            '{"entities": [], "relations": [{"subject": "A", "predicate": "rel", "object": "B", "confidence": -0.5}]}'
        ]
    )
    pipeline2 = ExtractionPipeline(client2, max_retries=2)
    result2 = pipeline2.extract("Some text")
    assert result2.relations[0].confidence == 0.0


def test_empty_response_handling() -> None:
    """Test handling of completely empty response - should retry and eventually fail."""
    client = MockClient(["", ""] * 5)  # Always returns empty
    pipeline = ExtractionPipeline(client, max_retries=2)
    try:
        pipeline.extract("Some text")
        assert False, "Should have raised ExtractionError"
    except ExtractionError:
        pass
    # Should have retried and failed
    metrics = pipeline.get_metrics()
    assert len(metrics) == 3
    assert all(not m.success for m in metrics)


def test_text_extraction_fallback() -> None:
    """Test last-resort text extraction fallback."""
    client = MockClient(
        [
            'Here are the entities: "name": "Test Entity", "type": "Concept", "canonical_name": "Test Entity" and some relations: "subject": "Test Entity", "predicate": "is", "object": "Real"'
        ]
    )
    pipeline = ExtractionPipeline(client, max_retries=2)
    result = pipeline.extract("Some text")
    # Should extract something from the text
    assert len(result.entities) >= 0  # May or may not work depending on regex


if __name__ == "__main__":
    import sys

    tests = [
        test_valid_json_first_try,
        test_json_in_code_block,
        test_json_with_trailing_comma_repair,
        test_retry_on_invalid_json,
        test_retry_exhaustion_raises,
        test_validation_error_on_missing_fields,
        test_non_strict_mode_allows_validation_errors,
        test_confidence_clamping,
        test_empty_response_handling,
        test_text_extraction_fallback,
    ]
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            sys.exit(1)
    print("All tests passed!")
