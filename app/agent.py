"""Agent orchestration. Routing is deterministic on top of the LLM's intent classification:
never recommend on a vague first turn, clarify at most until the turn budget tightens, and
build the recommendation SET deterministically (the LLM only writes the reply and fills any
leftover slots), so links can never be hallucinated."""
from __future__ import annotations

import json
import logging
import re

from .bundles import Bundles
from .catalog import Catalog
from .guardrails import Guard
from .llm_client import Llm, LlmError
from .prompts import (
    COMPARE_SYSTEM,
    COMPARE_USER,
    REPLY_SYSTEM,
    REPLY_USER,
    SELECT_SYSTEM,
    SELECT_USER,
)
from .retrieval import ANCHOR_IDS, Retriever
from .state import State, StateReader, fmt_history, user_text

log = logging.getLogger("agent")

MARKER = "Current shortlist:"
TARGET = 8
ANCHOR_PICK = ["occupational-personality-questionnaire-opq32r", "shl-verify-interactive-g"]

INJECT = (
    "I can only help with selecting SHL assessments from our catalog, so I can't follow that "
    "request. Tell me about the role you're hiring for and I'll suggest relevant assessments."
)
OFFTOPIC = (
    "I'm focused on recommending SHL assessments and can't help with that. If you share the role "
    "or skills you're hiring for, I'll suggest suitable assessments."
)
OPEN = "Happy to help you find the right SHL assessments. What role or skills are you hiring for?"


class Agent:
    def __init__(self, cat: Catalog, ret: Retriever, llm: Llm | None = None):
        self.cat = cat
        self.ret = ret
        self.llm = llm or Llm()
        self.state = StateReader(self.llm)
        self.bundles = Bundles(cat)

    # --- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _clean(msgs: list[dict]) -> list[dict]:
        out = []
        for m in msgs or []:
            content = m.get("content")
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            out.append({"role": m.get("role") or "user", "content": content})
        return out

    @staticmethod
    def _has_user(msgs: list[dict]) -> bool:
        return any((m.get("role") or "") == "user" for m in msgs)

    @staticmethod
    def _asked(msgs: list[dict]) -> bool:
        for m in reversed(msgs):
            if (m.get("role") or "") == "assistant":
                return m.get("content", "").rstrip().endswith("?")
        return False

    @staticmethod
    def _ask(st: State) -> str:
        c = st.cons or {}
        if not c.get("role_or_skill"):
            return "Happy to help narrow this down. What role or skills are you hiring for?"
        if not c.get("seniority"):
            return "Got it. What seniority level are you targeting — entry, mid, or senior?"
        return "Could you tell me a bit more about what you'd like the assessment to measure?"

    @staticmethod
    def _marker(recs: list[dict]) -> str:
        return f"{MARKER} " + "; ".join(r["name"] for r in recs)

    def _prior(self, msgs: list[dict]) -> list[dict]:
        # Resolve the most recent shortlist we emitted back into catalog items.
        for m in reversed(msgs):
            if (m.get("role") or "") != "assistant":
                continue
            content = m.get("content") or ""
            if MARKER not in content:
                continue
            line = content.split(MARKER, 1)[1].splitlines()[0]
            items: list[dict] = []
            seen: set[str] = set()
            for name in line.split(";"):
                it = self.cat.exact_by_name(name.strip())
                if it and it["id"] not in seen:
                    items.append(it)
                    seen.add(it["id"])
            return items
        return []

    def _anchors(self, recs: list[dict], excl: set[str]) -> list[dict]:
        urls = {r["url"] for r in recs}
        for aid in ANCHOR_PICK:
            if len(recs) >= 10:
                break
            it = self.cat.get(aid)
            if not it or it["url"] in urls:
                continue
            if excl and set((it.get("test_type") or "").split(",")) & excl:
                continue
            recs.append(Catalog.to_rec(it))
            urls.add(it["url"])
        return recs

    # --- deterministic floor -------------------------------------------------------------------
    def _floor(self, msgs: list[dict], excl: set[str] | None = None,
               limit: int = TARGET) -> list[dict]:
        """Measured-best no-LLM set, from raw user text: carried battery -> per-skill name
        matches -> diverse BM25 coverage -> flagship anchors. Shared by fallback and the main
        path, which only extends it."""
        excl = excl or set()
        q = user_text(msgs)
        cands = self.ret.pool(q, k=24, filters=None)
        prior = self._prior(msgs)

        recs: list[dict] = []
        urls: set[str] = set()
        fam: dict[str, int] = {}

        def take(it: dict | None, diverse: bool) -> None:
            if not it or len(recs) >= 10 or it["url"] in urls:
                return
            if excl and set((it.get("test_type") or "").split(",")) & excl:
                return
            toks = re.findall(r"[a-z0-9]+", it["name"].lower())
            key = toks[0] if toks else it["url"]
            if diverse and fam.get(key, 0) >= 2:
                return
            recs.append(Catalog.to_rec(it))
            urls.add(it["url"])
            fam[key] = fam.get(key, 0) + 1

        for it in prior:
            if len(recs) >= limit:
                break
            take(it, diverse=False)

        name_ids = self.ret.name_ids(q)
        promoted = self.bundles.promote(q)
        promoted_set = set(promoted)

        ordered_ids: list[str] = []
        for i in promoted + name_ids:
            if i not in ordered_ids:
                ordered_ids.append(i)
        seen_ids = set(ordered_ids)
        ordered = [self.cat.get(i) for i in ordered_ids if self.cat.get(i)] + [
            it for it in cands if it["id"] not in ANCHOR_IDS and it["id"] not in seen_ids
        ]
        for it in ordered:
            if len(recs) >= limit:
                break
            take(it, diverse=it["id"] not in promoted_set)

        return self._anchors(recs, excl)

    # --- public no-LLM fallback ----------------------------------------------------------------
    def fallback(self, msgs: list[dict]) -> dict:
        """Deterministic recommendation with no LLM calls, used on timeout/crash so a final
        turn is never empty."""
        msgs = self._clean(msgs)
        if not self._has_user(msgs):
            return {"reply": OPEN, "recommendations": [], "end_of_conversation": False}
        recs = Guard.validate(self._floor(msgs), self.cat)
        reply = "Here are SHL assessments that fit what you've described."
        if recs:
            reply = f"{reply}\n\n{self._marker(recs)}"
        return {"reply": reply, "recommendations": recs, "end_of_conversation": False}

    # --- recommend / refine --------------------------------------------------------------------
    def _recommend(self, st: State, msgs: list[dict]) -> tuple[str, list[dict]]:
        excl = set((st.cons or {}).get("test_types_exclude") or [])

        # The deterministic floor owns the SET. Build it first so we know whether any slot is
        # still open for the LLM to fill.
        recs = self._floor(msgs, excl=excl)

        if len(recs) >= 10:
            # No open slots: ask only for the natural-language reply with a tiny prompt. This
            # avoids the heavy candidate block, so the common case rarely hits a rate limit.
            reply = self._write_reply(msgs, recs)
        else:
            reply = self._select_fill(st, msgs, recs, excl)

        if not reply:
            reply = "Here are SHL assessments that fit what you've described."
        if recs:
            reply = f"{reply}\n\n{self._marker(recs)}"
        return reply, recs

    def _write_reply(self, msgs: list[dict], recs: list[dict]) -> str:
        try:
            raw = self.llm.chat(
                REPLY_SYSTEM,
                REPLY_USER.format(
                    history=fmt_history(msgs),
                    shortlist="; ".join(r["name"] for r in recs),
                ),
                model=self.llm.select_model,
                temp=0.3,
                max_tokens=200,
            )
            return str(raw.get("reply", "") or "").strip()
        except LlmError:
            log.warning("reply LLM call failed; using template")
            return ""

    def _select_fill(self, st: State, msgs: list[dict], recs: list[dict], excl: set[str]) -> str:
        # Floor left open slots: let the LLM both write the reply and fill the remainder from a
        # candidate block (it can never displace the floor, only extend it).
        q = " ".join(p for p in (user_text(msgs), st.query) if p).strip()
        cands = self.ret.pool(q, k=24, filters=st.cons)
        prior = self._prior(msgs)
        ids_in = {it["id"] for it in cands}
        for it in prior:
            if it["id"] not in ids_in:
                cands.append(it)
                ids_in.add(it["id"])

        block = "\n".join(
            f"- id={it['id']} | {it['name']} | type={it['test_type']} | "
            f"{(it.get('description') or '')[:80].replace(chr(10), ' ')}"
            for it in cands
        )
        prior_block = "; ".join(f"{it['id']} ({it['name']})" for it in prior) or "(none yet)"

        reply, ids = "", []
        try:
            raw = self.llm.chat(
                SELECT_SYSTEM,
                SELECT_USER.format(
                    history=fmt_history(msgs),
                    constraints=json.dumps(st.cons, ensure_ascii=False),
                    prior_shortlist=prior_block,
                    candidates=block,
                ),
                model=self.llm.select_model,
                temp=0.2,
                max_tokens=600,
            )
            reply = str(raw.get("reply", "") or "").strip()
            ids = [str(x) for x in (raw.get("ids") or [])]
        except LlmError:
            log.warning("select LLM call failed; using deterministic floor only")

        urls = {r["url"] for r in recs}
        for cid in ids:
            if len(recs) >= 10:
                break
            if cid not in ids_in:
                continue
            it = self.cat.get(cid)
            if not it or it["url"] in urls:
                continue
            if excl and set((it.get("test_type") or "").split(",")) & excl:
                continue
            recs.append(Catalog.to_rec(it))
            urls.add(it["url"])
        return reply

    # --- grounded compare ----------------------------------------------------------------------
    def _compare(self, st: State, msgs: list[dict]) -> tuple[str, list[dict]]:
        matched: list[dict] = []
        seen: set[str] = set()
        for name in st.compare:
            for it in self.cat.match_by_name(name, limit=1):
                if it["id"] not in seen:
                    matched.append(it)
                    seen.add(it["id"])

        if not matched:
            return (
                "I couldn't find those specific assessments in the SHL catalog. Could you confirm "
                "the exact names, or tell me the role so I can suggest options?",
                [],
            )

        facts = "\n".join(
            f"- {it['name']} | type={it['test_type']} | keys={','.join(it.get('keys', []))} | "
            f"duration={it.get('duration') or 'not listed'} | remote={it.get('remote')} | "
            f"adaptive={it.get('adaptive')} | description={it.get('description') or 'not listed'}"
            for it in matched
        )

        try:
            raw = self.llm.chat(
                COMPARE_SYSTEM,
                COMPARE_USER.format(history=fmt_history(msgs), facts=facts),
                temp=0.2,
                max_tokens=500,
            )
            reply = str(raw.get("reply", "") or "").strip()
        except LlmError:
            reply = f"Here is what the catalog lists for {' and '.join(it['name'] for it in matched)}."

        # Keep the established battery on a compare turn so a final turn doesn't collapse it.
        prior = self._prior(msgs)
        recs: list[dict] = []
        urls: set[str] = set()
        for it in prior + matched:
            if len(recs) >= 10:
                break
            rec = Catalog.to_rec(it)
            if rec["url"] in urls:
                continue
            recs.append(rec)
            urls.add(rec["url"])
        if recs:
            reply = f"{reply}\n\n{self._marker(recs)}"
        return reply, recs

    # --- entry point ---------------------------------------------------------------------------
    def handle(self, msgs: list[dict]) -> dict:
        msgs = self._clean(msgs)
        if not self._has_user(msgs):
            return {"reply": OPEN, "recommendations": [], "end_of_conversation": False}

        st = self.state.read(msgs)

        commit = (
            st.user_turns >= 2 or st.total >= 6 or self._asked(msgs)
        )
        role = (st.cons or {}).get("role_or_skill") or ""
        sig = [t for t in user_text(msgs).split() if len(t) > 2]
        concrete = bool(role.strip()) and len(sig) >= 5
        should = st.enough or concrete

        end = False
        if st.intent == "injection":
            reply, recs = INJECT, []
        elif st.intent == "off_topic":
            reply, recs = OFFTOPIC, []
        elif st.intent == "compare" and st.compare:
            reply, recs = self._compare(st, msgs)
        elif st.intent == "closing":
            reply, recs = self._recommend(st, msgs)
            reply = (
                (reply or "Glad that works — here's your final shortlist.") if recs
                else "Glad I could help. Let me know if you'd like to look at anything else."
            )
            end = True
        elif st.intent == "clarify" and not commit and not should:
            reply, recs = self._ask(st), []
        else:
            reply, recs = self._recommend(st, msgs)

        recs = Guard.validate(recs, self.cat)
        return {"reply": reply, "recommendations": recs, "end_of_conversation": end}
