"""Curated SHL product relationships that a lexical retriever cannot infer on its own.

Two kinds:
- Report families: reporting products derived from a flagship instrument (the OPQ reports,
  the Global Skills development report). The reference agent bundles these with the
  instrument, gated by the conversation theme.
- Companions: role/context tests the reference agent reliably adds for a given hiring
  situation (a spoken-language screen for contact centres, a dependability instrument for
  safety-critical roles, a numerical-reasoning test for graduate analysts, ...).

Each entry pairs a catalog id with keyword cues; when a cue appears in the conversation the
id is promoted into the shortlist ahead of generic BM25 filler and exempt from the diversity
cap, so these deliberate bundles are never crowded out."""
from __future__ import annotations

from .catalog import Catalog

SKILL_CUES = (
    "skill", "re-skill", "reskill", "upskill", "audit", "talent", "development", "competenc",
)

# flagship instrument -> its report products, each gated by theme cues (None = always).
REPORTS = {
    "occupational-personality-questionnaire-opq32r": [
        ("opq-universal-competency-report-2-0",
         ("lead", "executive", "cxo", "director", "senior", "competenc",
          "benchmark", "audit", "re-skill", "reskill")),
        ("opq-leadership-report", ("lead", "executive", "cxo", "director", "senior")),
        ("opq-mq-sales-report", ("sales", "selling", "revenue")),
    ],
    "global-skills-assessment": [
        ("global-skills-development-report", None),
    ],
}

# role/context companion tests, each gated by keyword cues.
COMPANIONS = [
    ("salestransformationreport2-0-individualcontributor",
     ("sales", "selling", "revenue", "seller")),
    ("dependability-and-safety-instrument-dsi",
     ("safety", "trust", "patient", "dependab", "security", "hipaa",
      "healthcare", "records", "compliance")),
    ("medical-terminology-new", ("medical", "healthcare", "patient", "clinical", "hipaa")),
    ("microsoft-word-365-essentials-new",
     ("patient records", "healthcare admin", "records", "hipaa")),
    ("basic-statistics-new",
     ("financial", "finance", "analyst", "statistic", "numerical", "quant")),
    ("smart-interview-live-coding",
     ("engineer", "developer", "coding", "programmer", "software", "backend", "infrastructure")),
    ("linux-programming-general",
     ("networking", "infrastructure", "systems", "backend", "kernel", "linux")),
    ("svar-spoken-english-us-new",
     ("contact cent", "call cent", "call centre", "inbound call", "contact centre")),
    ("shl-verify-interactive-numerical-reasoning",
     ("graduate", "numerical", "analyst", "campus", "final-year", "final year")),
]


class Bundles:
    def __init__(self, cat: Catalog):
        self.cat = cat

    def promote(self, q: str) -> list[str]:
        """Ids to seed into the shortlist for this conversation, deduped and in priority order."""
        ql = (q or "").lower()
        ids: list[str] = []

        flagships = ["occupational-personality-questionnaire-opq32r"]
        if any(cue in ql for cue in SKILL_CUES):
            ids.append("global-skills-assessment")
            flagships.append("global-skills-assessment")

        for fid in flagships:
            for rid, gate in REPORTS.get(fid, []):
                if gate is None or any(cue in ql for cue in gate):
                    ids.append(rid)

        for cid, gate in COMPANIONS:
            if any(cue in ql for cue in gate):
                ids.append(cid)

        out: list[str] = []
        seen: set[str] = set()
        for i in ids:
            if i not in seen and self.cat.get(i):
                seen.add(i)
                out.append(i)
        return out
