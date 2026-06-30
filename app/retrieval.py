"""BM25 lexical retrieval over the catalog plus a recall-oriented candidate pool.

BM25 (not embeddings): the catalog is ~380 short structured items whose matching signal is
lexical (skill names like Java, SQL, OPQ, Docker), and BM25 beat a small semantic index here.
The pool seeds a few SHL flagship anchors and their report siblings, which are related but
lexically far from the query, so BM25 alone misses them. Recall@10 has no precision penalty,
so seeding them can only help."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .catalog import Catalog

ANCHOR_IDS = [
    "occupational-personality-questionnaire-opq32r",
    "shl-verify-interactive-g",
    "graduate-scenarios",
    "global-skills-assessment",
]

FAMILIES = {
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

PREFIX_STOP = {
    "new", "the", "a", "an", "of", "and", "for", "shl", "test", "assessment",
    "report", "level", "general", "based", "solution",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def sig_tokens(name: str) -> list[str]:
    return [t for t in tokenize(name) if t not in PREFIX_STOP]


class Retriever:
    def __init__(self, cat: Catalog):
        self.cat = cat
        self.bm25 = BM25Okapi([tokenize(self._doc(it)) for it in cat.items])
        self._prefix = self._index_prefix()
        self._names = self._index_names()

    @staticmethod
    def _doc(it: dict) -> str:
        # Name repeated to weight it; it carries the strongest matching signal.
        keys = " ".join(it.get("keys", []))
        levels = " ".join(it.get("job_levels", []))
        return " ".join([it["name"]] * 3 + [it.get("description", ""), keys, keys, levels])

    def _index_names(self) -> dict[str, list[str]]:
        # token -> ids whose name contains it. Lets us pull the precise single-skill test
        # for each concrete term the user names, which a long diluted query buries in BM25.
        idx: dict[str, list[str]] = {}
        for it in self.cat.items:
            for tok in set(sig_tokens(it["name"])):
                if len(tok) >= 3:
                    idx.setdefault(tok, []).append(it["id"])
        return {tok: ids for tok, ids in idx.items() if len(ids) <= 25}

    def _name_hits(self, q: str, per_token: int = 4, cap: int = 22) -> list[str]:
        toks = tokenize(q)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        score_by_id = {it["id"]: scores[i] for i, it in enumerate(self.cat.items)}
        qt = [t for t in dict.fromkeys(toks) if len(t) >= 3]
        out: list[str] = []
        seen: set[str] = set()
        for tok in qt:
            ids = self._names.get(tok)
            if not ids:
                continue
            by_score = sorted(ids, key=lambda i: -score_by_id.get(i, 0.0))[:per_token]
            by_len = sorted(ids, key=lambda i: len(self.cat.get(i)["name"]))[:2]
            for cid in dict.fromkeys(by_score + by_len):
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
                if len(out) >= cap:
                    return out
        return out

    def name_ids(self, q: str) -> list[str]:
        return self._name_hits(q)

    def _index_prefix(self) -> dict[tuple, list[str]]:
        # first two significant name tokens -> ids, for report-variant sibling lookup.
        idx: dict[tuple, list[str]] = {}
        for it in self.cat.items:
            sig = sig_tokens(it["name"])
            if len(sig) >= 2:
                idx.setdefault((sig[0], sig[1]), []).append(it["id"])
        return idx

    def _siblings(self, cid: str, cap: int = 4) -> list[str]:
        it = self.cat.get(cid)
        if not it:
            return []
        sig = sig_tokens(it["name"])
        if len(sig) < 2:
            return []
        return [s for s in self._prefix.get((sig[0], sig[1]), []) if s != cid][:cap]

    def search(self, q: str, k: int = 40, filters: dict | None = None) -> list[dict]:
        # Wide BM25 ranking with only hard test-type exclusions; over-filtering kills recall.
        filters = filters or {}
        q = (q or "").strip()
        if not q:
            order = list(range(len(self.cat.items)))
        else:
            scores = self.bm25.get_scores(tokenize(q))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
        excl = set(filters.get("test_types_exclude") or [])
        out: list[dict] = []
        for i in order:
            it = self.cat.items[i]
            if excl and set((it.get("test_type") or "").split(",")) & excl:
                continue
            out.append(it)
            if len(out) >= k:
                break
        return out

    def pool(self, q: str, k: int = 26, filters: dict | None = None,
             anchors: bool = True) -> list[dict]:
        """Candidate set for the selector: BM25 top-k + per-skill name matches + flagship
        anchors and their report families, deduped and honoring hard exclusions."""
        filters = filters or {}
        excl = set(filters.get("test_types_exclude") or [])
        seen: set[str] = set()
        out: list[dict] = []

        def add(it: dict | None) -> None:
            if not it or it["id"] in seen:
                return
            if excl and set((it.get("test_type") or "").split(",")) & excl:
                return
            seen.add(it["id"])
            out.append(it)

        for it in self.search(q, k=k, filters=filters):
            add(it)
        for cid in self._name_hits(q):
            add(self.cat.get(cid))
        if anchors:
            for aid in ANCHOR_IDS:
                add(self.cat.get(aid))
            for aid in ANCHOR_IDS:
                for fid in FAMILIES.get(aid, []):
                    add(self.cat.get(fid))
                for sid in self._siblings(aid):
                    add(self.cat.get(sid))
        return out
