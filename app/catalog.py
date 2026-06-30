"""Catalog loading, id/name lookup, and grounded fuzzy matching for the compare behavior.

The catalog is the single source of truth for every name/url/test_type the service ever emits.
Nothing the LLM generates as free text is trusted for these fields.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

_STOPWORDS = {"the", "a", "an", "of", "and", "for", "new", "test", "assessment", "shl"}


def _norm_name(name: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {t for t in toks if t not in _STOPWORDS}


class Catalog:
    def __init__(self, items: list[dict]):
        self.items = items
        self.by_id = {it["id"]: it for it in items}
        self._by_name = {it["name"].strip(): it for it in items}
        self._name_tokens = [(it, _norm_name(it["name"])) for it in items]

    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> dict | None:
        return self.by_id.get(item_id)

    def exact_by_name(self, name: str) -> dict | None:
        """Exact-name lookup. Used to resolve a shortlist we previously emitted (we control the
        formatting, so exact match is reliable even for names containing odd unicode)."""
        return self._by_name.get((name or "").strip())

    @staticmethod
    def to_recommendation(item: dict) -> dict:
        """Project a catalog entry onto the exact response schema fields."""
        return {
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"],
        }

    def has_url(self, url: str) -> bool:
        return any(it["url"] == url for it in self.items)

    def match_by_name(self, query: str, limit: int = 3, threshold: float = 0.34) -> list[dict]:
        """Best-effort entity resolution for compare ("OPQ" -> OPQ32r).

        Combines token overlap (handles acronym/long-form drift) with a difflib ratio
        (handles spelling variants). Returns catalog items ordered by confidence.
        """
        q_tokens = _norm_name(query)
        q_lower = (query or "").lower().strip()
        if not q_lower:
            return []
        scored: list[tuple[float, dict]] = []
        for it, name_tokens in self._name_tokens:
            name_lower = it["name"].lower()
            overlap = 0.0
            if q_tokens and name_tokens:
                overlap = len(q_tokens & name_tokens) / len(q_tokens)
            ratio = SequenceMatcher(None, q_lower, name_lower).ratio()
            substring = 1.0 if (q_lower in name_lower or name_lower in q_lower) else 0.0
            score = max(overlap, ratio, substring)
            if score >= threshold:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:limit]]


def load_catalog(path: Path | None = None) -> Catalog:
    p = path or CATALOG_PATH
    items = json.loads(p.read_text(encoding="utf-8"), strict=False)
    return Catalog(items)
