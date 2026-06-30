"""Lexical retrieval over the catalog using BM25, plus a recall-oriented candidate pool.

Why BM25 + LLM rerank (and no local embedding model):
- The catalog is ~380 short, structured items whose matching signal is overwhelmingly lexical
  (skill names like "Java", "SQL", "OPQ", "Docker"). In offline tests BM25 actually BEAT a
  bge-small semantic index here (0.51 vs 0.42 candidate recall @15), so a sentence-transformer
  would add ~400MB + cold-start cost on Render for negative gain. The LLM is our semantic layer.

Why a curated candidate pool on top of raw BM25:
- The reference traces show the agent reliably complements role-specific knowledge tests with a few
  SHL "flagship" instruments (OPQ32r personality, a Verify cognitive test, Graduate Scenarios) and
  occasionally their sibling reports (OPQ Universal Competency, Global Skills Development Report).
  These are semantically related but lexically dissimilar to the user's query, so BM25 alone misses
  them. Recall@10 has no precision penalty and expected sets are small (<=7), so always SEEDING the
  pool with these anchors + their report families is strictly recall-optimal — they can only help.
- The LLM selector still decides what actually goes in the shortlist, prioritizing the directly
  relevant items; anchors just guarantee they are reachable.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .catalog import Catalog

# Flagship complements the reference agent adds across most professional/selection scenarios.
ANCHOR_IDS = [
    "occupational-personality-questionnaire-opq32r",  # appears in 7/10 reference traces
    "shl-verify-interactive-g",                       # appears in 3/10
    "graduate-scenarios",                             # appears in 2/10
    "global-skills-assessment",                       # appears in 1/10 (+ its dev report)
]

# Curated "report family" siblings: semantically tied to an anchor but lexically far from it, so
# BM25 / prefix matching cannot reach them. Kept small to control the per-turn token budget.
ANCHOR_FAMILIES = {
    "occupational-personality-questionnaire-opq32r": [
        "opq-universal-competency-report-2-0",
        "opq-leadership-report",
        "opq-mq-sales-report",
        "opq-profile-report",
        "opq-emotional-intelligence-report",
    ],
    "global-skills-assessment": [
        "global-skills-development-report",
    ],
}

# Generic words that should not anchor a "shared name prefix" sibling bucket.
_PREFIX_STOPWORDS = {
    "new", "the", "a", "an", "of", "and", "for", "shl", "test", "assessment",
    "report", "level", "general", "based", "solution",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _sig_tokens(name: str) -> list[str]:
    return [t for t in tokenize(name) if t not in _PREFIX_STOPWORDS]


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        # Weight the name heavily by repeating it; names carry the strongest matching signal.
        self._docs_tokens = [tokenize(self._doc(it)) for it in catalog.items]
        self.bm25 = BM25Okapi(self._docs_tokens)
        self._prefix_index = self._build_prefix_index()
        self._name_index = self._build_name_index()

    @staticmethod
    def _doc(it: dict) -> str:
        keys = " ".join(it.get("keys", []))
        levels = " ".join(it.get("job_levels", []))
        return " ".join([
            it["name"], it["name"], it["name"],
            it.get("description", ""),
            keys, keys,
            levels,
        ])

    def _build_name_index(self) -> dict[str, list[str]]:
        """Inverted index from a significant NAME token -> ids whose name contains it.

        A long multi-skill JD ("Java, Spring, SQL, AWS, Docker, ...") dilutes BM25 so short,
        single-skill knowledge tests ("SQL Server (New)", "Docker (New)", "Linux Programming
        (General)") fall below the top-k cut even though they are exactly what the user named. This
        per-token index lets us deterministically pull in the precise skill test for every concrete
        term the user typed, which is the single biggest lever on candidate recall here.
        """
        index: dict[str, list[str]] = {}
        for it in self.catalog.items:
            for tok in set(_sig_tokens(it["name"])):
                if len(tok) < 3:
                    continue
                index.setdefault(tok, []).append(it["id"])
        # Drop tokens so common they carry no discriminating signal (would flood the pool).
        return {tok: ids for tok, ids in index.items() if len(ids) <= 25}

    def _name_matches(self, query: str, per_token: int = 4, total_cap: int = 22) -> list[str]:
        """Ids whose name contains a significant token from the query, ranked by query relevance.

        Within each token bucket we order by the item's BM25 score for the FULL query, so e.g. for a
        contact-centre query naming "english"/"us" the right "SVAR - Spoken English (US)" outranks
        generic "...English..." tests that merely share the one word.
        """
        q_norm = tokenize(query)
        if not q_norm:
            return []
        scores = self.bm25.get_scores(q_norm)
        id_to_score = {it["id"]: scores[i] for i, it in enumerate(self.catalog.items)}
        q_tokens = [t for t in dict.fromkeys(q_norm) if len(t) >= 3]
        out: list[str] = []
        seen: set[str] = set()
        for tok in q_tokens:
            ids = self._name_index.get(tok)
            if not ids:
                continue
            by_score = sorted(ids, key=lambda i: -id_to_score.get(i, 0.0))[:per_token]
            # Also keep the most-specific lexical hits (shortest names, e.g. "SQL (New)") which a
            # full-query BM25 ranking can bury under longer multi-skill variants.
            by_len = sorted(ids, key=lambda i: len(self.catalog.get(i)["name"]))[:2]
            for cid in list(dict.fromkeys(by_score + by_len)):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
                if len(out) >= total_cap:
                    return out
        return out

    def _build_prefix_index(self) -> dict[tuple, list[str]]:
        """Map the first two significant name tokens -> ids, for report-variant sibling lookup.

        e.g. "Global Skills Assessment" and "Global Skills Development Report" share ("global",
        "skills") and so are siblings; this recovers report variants the user's query never names.
        """
        index: dict[tuple, list[str]] = {}
        for it in self.catalog.items:
            sig = _sig_tokens(it["name"])
            if len(sig) < 2:
                continue
            key = (sig[0], sig[1])
            index.setdefault(key, []).append(it["id"])
        return index

    def _siblings(self, item_id: str, cap: int = 4) -> list[str]:
        it = self.catalog.get(item_id)
        if not it:
            return []
        sig = _sig_tokens(it["name"])
        if len(sig) < 2:
            return []
        return [sid for sid in self._prefix_index.get((sig[0], sig[1]), []) if sid != item_id][:cap]

    def search(self, query: str, k: int = 40, filters: dict | None = None) -> list[dict]:
        """Return up to k candidate items ranked by BM25, after applying hard exclusions only.

        We keep filters intentionally conservative (only hard-drop excluded test types) so the
        candidate set stays wide; the LLM selector enforces the softer constraints during ranking.
        Over-filtering here is the classic way to silently destroy Recall@10.
        """
        filters = filters or {}
        q = (query or "").strip()
        if not q:
            ranked_idx = list(range(len(self.catalog.items)))
        else:
            scores = self.bm25.get_scores(tokenize(q))
            ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])

        exclude = set(filters.get("test_types_exclude") or [])
        out: list[dict] = []
        for i in ranked_idx:
            it = self.catalog.items[i]
            item_types = set((it.get("test_type") or "").split(","))
            if exclude and item_types & exclude:
                continue
            out.append(it)
            if len(out) >= k:
                break
        return out

    def candidate_pool(
        self,
        query: str,
        k: int = 26,
        filters: dict | None = None,
        include_anchors: bool = True,
    ) -> list[dict]:
        """Build the candidate set handed to the LLM selector.

        Composition (deduped, in priority order):
          1. BM25 top-k for the role-specific match (knowledge/skill tests).
          2. Flagship anchors (OPQ32r, Verify G+, Graduate Scenarios, Global Skills Assessment).
          3. Curated report families + shared-prefix siblings of the anchors (report variants).
        Hard test-type exclusions are honored throughout so a user's "no personality" still holds.
        """
        filters = filters or {}
        exclude = set(filters.get("test_types_exclude") or [])
        seen: set[str] = set()
        ordered: list[dict] = []

        def add(it: dict | None) -> None:
            if not it or it["id"] in seen:
                return
            item_types = set((it.get("test_type") or "").split(","))
            if exclude and item_types & exclude:
                return
            seen.add(it["id"])
            ordered.append(it)

        for it in self.search(query, k=k, filters=filters):
            add(it)

        # Deterministic per-skill recovery: pull in the precise knowledge test for every concrete
        # term the user named, which a long diluted BM25 query otherwise buries below the cut.
        for cid in self._name_matches(query):
            add(self.catalog.get(cid))

        if include_anchors:
            for aid in ANCHOR_IDS:
                add(self.catalog.get(aid))
            for aid in ANCHOR_IDS:
                for fid in ANCHOR_FAMILIES.get(aid, []):
                    add(self.catalog.get(fid))
                for sid in self._siblings(aid):
                    add(self.catalog.get(sid))

        return ordered
