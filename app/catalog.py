"""Catalog loading, lookup, and fuzzy name matching. The catalog is the only source
of truth for the name/url/test_type the service emits."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

STOP = {"the", "a", "an", "of", "and", "for", "new", "test", "assessment", "shl"}


def name_tokens(name: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {t for t in toks if t not in STOP}


class Catalog:
    def __init__(self, items: list[dict]):
        self.items = items
        for it in items:
            # Collapse stray whitespace/newlines in names so the shortlist marker (newline
            # delimited) survives carry-forward and we never emit a multi-line name.
            it["name"] = re.sub(r"\s+", " ", it["name"]).strip()
        self.by_id = {it["id"]: it for it in items}
        self._by_name = {it["name"].strip(): it for it in items}
        self._tokens = [(it, name_tokens(it["name"])) for it in items]

    def __len__(self) -> int:
        return len(self.items)

    def get(self, cid: str) -> dict | None:
        return self.by_id.get(cid)

    def exact_by_name(self, name: str) -> dict | None:
        # Resolves a shortlist we emitted ourselves, so exact match is reliable.
        return self._by_name.get((name or "").strip())

    @staticmethod
    def to_rec(it: dict) -> dict:
        return {"name": it["name"], "url": it["url"], "test_type": it["test_type"]}

    def has_url(self, url: str) -> bool:
        return any(it["url"] == url for it in self.items)

    def match_by_name(self, q: str, limit: int = 3, threshold: float = 0.34) -> list[dict]:
        """Resolve a free-text name to catalog items (e.g. "OPQ" -> OPQ32r) using token
        overlap plus a difflib ratio, ranked by confidence."""
        qt = name_tokens(q)
        ql = (q or "").lower().strip()
        if not ql:
            return []
        scored: list[tuple[float, dict]] = []
        for it, nt in self._tokens:
            nl = it["name"].lower()
            overlap = len(qt & nt) / len(qt) if (qt and nt) else 0.0
            ratio = SequenceMatcher(None, ql, nl).ratio()
            sub = 1.0 if (ql in nl or nl in ql) else 0.0
            score = max(overlap, ratio, sub)
            if score >= threshold:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:limit]]


def load_catalog(path: Path | None = None) -> Catalog:
    p = path or CATALOG_PATH
    return Catalog(json.loads(p.read_text(encoding="utf-8"), strict=False))
