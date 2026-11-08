from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    data_dir: Path
    request_timeout: float
    max_retries: int

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_settings(env_path: Path | None = None) -> Settings:
    load_dotenv(env_path or Path(".env"))
    return Settings(
        llm_base_url=os.environ.get(
            "GRAPHRAG_LLM_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/"),
        llm_api_key=os.environ.get("GRAPHRAG_LLM_API_KEY", ""),
        llm_model=os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4o-mini"),
        data_dir=Path(os.environ.get("GRAPHRAG_DATA_DIR", "./data")),
        request_timeout=float(os.environ.get("GRAPHRAG_REQUEST_TIMEOUT", "60")),
        max_retries=int(os.environ.get("GRAPHRAG_MAX_RETRIES", "2")),
    )
