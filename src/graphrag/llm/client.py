from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    """Raised when rate limited (HTTP 429)."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(slots=True)
class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using stdlib HTTP (0 extra deps)."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    max_retries: int = 5  # Increased for rate limits

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
            "User-Agent": "python-urllib/3.12",
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
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429:
                    # Rate limited - use Retry-After header or exponential backoff
                    retry_after = self._get_retry_after(exc)
                    wait_time = retry_after if retry_after is not None else self._exponential_backoff(attempt)
                    print(f"Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait_time)
                    continue
                # Other HTTP errors - retry with backoff
                if attempt >= self.max_retries:
                    break
                wait_time = self._exponential_backoff(attempt)
                time.sleep(wait_time)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                wait_time = self._exponential_backoff(attempt)
                time.sleep(wait_time)
        raise LLMError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}") from last_error

    def _get_retry_after(self, exc: urllib.error.HTTPError) -> float | None:
        """Extract Retry-After header (seconds or HTTP date)."""
        retry_after = exc.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            # Could be seconds (int) or HTTP date
            return float(retry_after)
        except ValueError:
            # HTTP date format - parse it
            try:
                import email.utils
                retry_date = email.utils.parsedate_to_datetime(retry_after)
                return max(0.0, (retry_date - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                return None

    def _exponential_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^attempt + random jitter."""
        base = 2.0
        max_wait = 60.0
        wait = min(base * (2 ** attempt) + random.uniform(0, 1), max_wait)
        return wait

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {body!r}") from exc
