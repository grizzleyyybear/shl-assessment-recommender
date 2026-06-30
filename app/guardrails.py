"""Deterministic guardrails: cheap input heuristics plus an output validator that drops
any recommendation not backed by the catalog and clamps the list to 10."""
from __future__ import annotations

import re

from .catalog import Catalog

INJECTION = [
    r"ignore (all |the |your |previous |above )+instructions",
    r"disregard (all|the|your|previous|above)",
    r"you are now",
    r"forget (everything|your|the)",
    r"reveal (your |the )?(system )?prompt",
    r"system prompt",
    r"act as",
    r"jailbreak",
    r"regardless of (fit|relevance|the catalog)",
    r"just say yes",
    r"pretend (you|to)",
]

OFFTOPIC = [
    r"\bwhat should i wear\b",
    r"\binterview (tips|questions|advice)\b",
    r"\bsalary\b",
    r"\bnegotiat",
    r"\bvisa\b",
    r"\blegal\b",
    r"\blawsuit\b",
    r"\bdiscriminat",
    r"\bfire (an|my|the) employee",
    r"\bwrite (me )?(a|an) (poem|essay|email|cover letter)\b",
]


class Guard:
    @staticmethod
    def is_injection(text: str) -> bool:
        t = (text or "").lower()
        return any(re.search(p, t) for p in INJECTION)

    @staticmethod
    def is_offtopic(text: str) -> bool:
        t = (text or "").lower()
        return any(re.search(p, t) for p in OFFTOPIC)

    @staticmethod
    def validate(recs: list[dict], cat: Catalog) -> list[dict]:
        """Keep only recs whose url maps to a real catalog entry; dedupe; clamp to 10."""
        out: list[dict] = []
        seen: set[str] = set()
        for r in recs:
            url = (r or {}).get("url", "")
            if url in seen or not cat.has_url(url):
                continue
            out.append({
                "name": r.get("name", ""),
                "url": url,
                "test_type": r.get("test_type", ""),
            })
            seen.add(url)
            if len(out) >= 10:
                break
        return out
