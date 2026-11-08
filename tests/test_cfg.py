from pathlib import Path

from graphrag.config import get_settings


def test_get_settings_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GRAPHRAG_LLM_API_KEY", raising=False)
    monkeypatch.setenv("GRAPHRAG_DATA_DIR", str(tmp_path / "data"))
    s = get_settings(env_path=tmp_path / "missing.env")
    assert s.llm_model
    assert s.raw_dir == tmp_path / "data" / "raw"
    assert s.llm_api_key == ""
