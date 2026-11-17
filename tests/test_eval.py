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
