"""Small Groq wrapper: JSON-mode chat with a hard timeout, a short retry, and a typed
error so every caller can fall back deterministically."""
from __future__ import annotations

import json
import os
import time

from groq import Groq


class LlmError(Exception):
    pass


class Llm:
    def __init__(self) -> None:
        # Both calls default to the 8B model: on Groq's free tier it has far more daily
        # headroom than 70B, and the BM25+anchor pool carries the retrieval quality.
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.state_model = os.getenv("GROQ_STATE_MODEL", "llama-3.1-8b-instant")
        self.select_model = os.getenv("GROQ_SELECT_MODEL", self.model)
        self.timeout = float(os.getenv("GROQ_TIMEOUT", "10"))
        self.tries = int(os.getenv("GROQ_MAX_ATTEMPTS", "2"))
        self._cli: Groq | None = None

    def _client(self) -> Groq:
        if self._cli is None:
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise LlmError("GROQ_API_KEY is not set")
            self._cli = Groq(api_key=key, timeout=self.timeout, max_retries=0)
        return self._cli

    @staticmethod
    def _rate_limited(exc: Exception) -> bool:
        return getattr(exc, "status_code", None) == 429 or "rate" in str(exc).lower()

    def chat(self, sys: str, usr: str, *, model: str | None = None,
             temp: float = 0.2, max_tokens: int = 1024) -> dict:
        cli = self._client()
        mdl = model or self.model
        last: Exception | None = None
        for i in range(self.tries):
            try:
                resp = cli.chat.completions.create(
                    model=mdl,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": usr},
                    ],
                    temperature=temp,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content or "{}")
            except Exception as exc:  # normalize every failure to LlmError
                last = exc
                if i < self.tries - 1:
                    time.sleep((1.5 if self._rate_limited(exc) else 0.8) * (i + 1))
        raise LlmError(str(last))
