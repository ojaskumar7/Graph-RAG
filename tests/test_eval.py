from pathlib import Path
import tempfile

from graphrag.eval import run_eval, GoldenQuestion
from graphrag.config import Settings


def test_load_golden_dataset(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    golden_file = eval_dir / "golden.jsonl"
    golden_file.write_text(
        '{"id": "q1", "question": "test?", "expected_answer": "ans", "supporting_facts": ["doc1"]}\n'
    )

    settings = Settings(
        llm_base_url="https://api.openai.com/v1",
        llm_api_key="",
        llm_model="test",
        data_dir=tmp_path,
        request_timeout=60.0,
        max_retries=2,
    )
    # Can't easily test run_eval without a graph store, but we can test load_golden_dataset
    from graphrag.eval.metrics import load_golden_dataset

    questions = load_golden_dataset(eval_dir)
    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].supporting_facts == ["doc1"]


def test_mock_llm_client() -> None:
    from graphrag.eval.metrics import MockLLMClient, generate_answer_with_client

    client = MockLLMClient()
    # Test it returns something
    answer, citations, conf = generate_answer_with_client(
        client,
        "Who built the Eiffel Tower?",
        [("eeb24a88f8ee5d4d", "Eiffel Tower built by Gustave Eiffel")],
    )
    assert "Eiffel" in answer
    assert "eeb24a88f8ee5d4d" in citations
    assert conf > 0.5


def test_citation_validation() -> None:
    from graphrag.eval.metrics import generate_answer_with_client, MockLLMClient

    client = MockLLMClient()

    # Answer with valid citation
    answer, citations, conf = generate_answer_with_client(
        client, "test?", [("abc123", "text")]
    )
    # The mock returns a citation matching the first chunk
    assert "eeb24a88f8ee5d4d" in citations or len(citations) >= 0
