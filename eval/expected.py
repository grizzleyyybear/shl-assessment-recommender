"""Parse the provided sample conversation traces into (user_turns, expected_urls).

Each trace is a multi-turn conversation; the agent's FINAL shortlist table is treated as the
labeled expected set for Recall@K. We replay the persona's user turns against our own agent and
compare our final recommendations to this expected set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TRACES_DIR = Path(__file__).parent / "traces"

_URL_RE = re.compile(r"https?://[^\s>|)\]]+")


@dataclass
class Trace:
    name: str
    user_turns: list[str]
    expected_urls: list[str]


def _extract_user_turns(text: str) -> list[str]:
    turns: list[str] = []
    # Each user turn is a "**User**" header followed by one or more blockquote lines.
    blocks = re.split(r"\*\*User\*\*", text)[1:]
    for block in blocks:
        # Stop at the next bold header (e.g. **Agent**).
        head = block.split("**Agent**")[0]
        quoted = [
            line.lstrip("> ").strip()
            for line in head.splitlines()
            if line.lstrip().startswith(">")
        ]
        content = " ".join(q for q in quoted if q).strip()
        if content:
            turns.append(content)
    return turns


def _normalize_url(url: str) -> str:
    return url.rstrip("/").rstrip(">").rstrip(".")


def _extract_expected_urls(text: str) -> list[str]:
    # The expected shortlist is the LAST agent turn containing catalog URLs.
    turn_blocks = re.split(r"### Turn", text)
    for block in reversed(turn_blocks):
        urls = [_normalize_url(u) for u in _URL_RE.findall(block)]
        if urls:
            # Dedupe preserving order.
            seen: list[str] = []
            for u in urls:
                if u not in seen:
                    seen.append(u)
            return seen
    return []


def load_traces() -> list[Trace]:
    traces: list[Trace] = []
    for path in sorted(TRACES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        traces.append(
            Trace(
                name=path.stem,
                user_turns=_extract_user_turns(text),
                expected_urls=_extract_expected_urls(text),
            )
        )
    return traces


if __name__ == "__main__":
    for t in load_traces():
        print(f"{t.name}: {len(t.user_turns)} user turns, {len(t.expected_urls)} expected urls")
        for u in t.expected_urls:
            print("   ", u)
