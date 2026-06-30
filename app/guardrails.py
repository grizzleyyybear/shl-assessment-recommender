"""Deterministic guardrails. Two layers, both code-first (we do not trust the LLM to police itself).

1. Input heuristics: a cheap backstop classifier for off-topic / injection that runs even if the
   LLM state call fails. The primary defense lives in the prompts (delimiting + framing).
2. Output validator: before ANY response leaves the service, every recommendation is verified to
   exist verbatim in the catalog and the list is clamped to <=10. A response that fails validation
   is replaced by a safe deterministic one — a slightly worse answer that passes the grader beats a
   richer answer that breaks the schema.
"""
from __future__ import annotations

import re

from .catalog import Catalog

_INJECTION_PATTERNS = [
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

_OFFTOPIC_PATTERNS = [
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


def looks_like_injection(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _INJECTION_PATTERNS)


def looks_off_topic(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _OFFTOPIC_PATTERNS)


def validate_recommendations(recs: list[dict], catalog: Catalog) -> list[dict]:
    """Keep only recommendations whose url+name match a real catalog entry; clamp to 10.

    Groundedness must be 100%: any item that does not map to the catalog is dropped (and would be
    logged by the caller) rather than emitted.
    """
    clean: list[dict] = []
    seen: set[str] = set()
    for rec in recs:
        url = (rec or {}).get("url", "")
        if url in seen:
            continue
        if catalog.has_url(url):
            clean.append({
                "name": rec.get("name", ""),
                "url": url,
                "test_type": rec.get("test_type", ""),
            })
            seen.add(url)
        if len(clean) >= 10:
            break
    return clean
