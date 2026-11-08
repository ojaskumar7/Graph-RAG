from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using stdlib HTTP (0 extra deps)."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    max_retries: int = 2

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise LLMError("GRAPHRAG_LLM_API_KEY is not set")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=data, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._extract_content(body)
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))
        raise LLMError(f"LLM request failed: {last_error}") from last_error

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {body!r}") from exc
