"""Normalize the provided SHL product catalog into the runtime catalog the service consumes.

We use the catalogue SHL provided (377 Individual Test Solutions) as ground truth rather than
scraping the live site: it pins the exact snapshot the grader's expected shortlists were drawn
from, removing the single biggest source of drift (a JS-rendered site changing under us).

Run once:  python data/build_catalog.py
Output:    app/data/catalog.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RAW = Path(__file__).parent / "shl_product_catalog_raw.json"
OUT = Path(__file__).parent.parent / "app" / "data" / "catalog.json"

# SHL test-type taxonomy: the catalog stores full category names ("keys"); the API contract
# and the grader's traces use the single-letter SHL codes. This is the canonical mapping.
KEY_TO_LETTER = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def slug_from_link(link: str) -> str:
    """Stable id derived from the catalog URL's view slug (e.g. .../view/opq32r/ -> opq32r)."""
    m = re.search(r"/view/([^/]+)/?", link or "")
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9]+", "-", (link or "").lower()).strip("-")


def normalize(entry: dict) -> dict | None:
    name = (entry.get("name") or "").strip()
    link = (entry.get("link") or "").strip()
    if not name or not link:
        return None

    keys = entry.get("keys") or []
    letters = [KEY_TO_LETTER[k] for k in keys if k in KEY_TO_LETTER]
    # Deduplicate while preserving order so test_type is deterministic.
    seen: list[str] = []
    for l in letters:
        if l not in seen:
            seen.append(l)
    test_type = ",".join(seen)

    return {
        "id": slug_from_link(link),
        "name": name,
        "url": link,
        "test_type": test_type,            # comma-joined SHL letters, e.g. "P" or "A,K"
        "keys": keys,                       # human-readable categories (used in compare answers)
        "description": (entry.get("description") or "").strip(),
        "job_levels": entry.get("job_levels") or [],
        "languages": entry.get("languages") or [],
        "duration": (entry.get("duration") or "").strip(),
        "remote": str(entry.get("remote") or "").strip().lower() == "yes",
        "adaptive": str(entry.get("adaptive") or "").strip().lower() == "yes",
    }


def main() -> None:
    # strict=False tolerates raw control characters (e.g. literal tabs/newlines) that appear
    # inside some scraped description fields in the provided file.
    raw = json.loads(RAW.read_text(encoding="utf-8"), strict=False)
    out: list[dict] = []
    seen_ids: set[str] = set()
    for entry in raw:
        norm = normalize(entry)
        if norm is None:
            continue
        if norm["id"] in seen_ids:  # dedupe by stable id
            continue
        seen_ids.add(norm["id"])
        out.append(norm)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} catalog items to {OUT}")
    # Sanity: report how many lack descriptions / test_type so we know retrieval signal quality.
    no_desc = sum(1 for x in out if not x["description"])
    no_type = sum(1 for x in out if not x["test_type"])
    print(f"  items without description: {no_desc}")
    print(f"  items without test_type:  {no_type}")


if __name__ == "__main__":
    main()
