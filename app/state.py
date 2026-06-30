"""Stateless state derivation: every /chat call re-derives all constraints from the full
history, so "refine" needs no extra machinery."""
from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import Guard
from .llm_client import Llm, LlmError
from .prompts import STATE_SYSTEM, STATE_USER

INTENTS = {"recommend", "clarify", "refine", "compare", "off_topic", "injection", "closing"}


@dataclass
class State:
    intent: str = "clarify"
    enough: bool = False
    cons: dict = field(default_factory=dict)
    query: str = ""
    compare: list[str] = field(default_factory=list)
    user_turns: int = 0
    total: int = 0
    asst_turns: int = 0


def fmt_history(msgs: list[dict]) -> str:
    return "\n".join(
        f"{(m.get('role') or 'user').upper()}: {m.get('content', '')}" for m in msgs
    )


def user_text(msgs: list[dict]) -> str:
    return " ".join(
        m.get("content", "") for m in msgs if (m.get("role") or "") == "user"
    ).strip()


class StateReader:
    def __init__(self, llm: Llm):
        self.llm = llm

    def read(self, msgs: list[dict]) -> State:
        users = sum(1 for m in msgs if (m.get("role") or "") == "user")
        assts = sum(1 for m in msgs if (m.get("role") or "") == "assistant")
        total = len(msgs)
        try:
            raw = self.llm.chat(
                STATE_SYSTEM,
                STATE_USER.format(history=fmt_history(msgs)),
                model=self.llm.state_model,
                temp=0.0,
                max_tokens=600,
            )
        except LlmError:
            st = self._fallback(msgs, users)
            st.asst_turns, st.total = assts, total
            return st

        intent = raw.get("intent", "clarify")
        if intent not in INTENTS:
            intent = "clarify"
        c = raw.get("constraints") or {}
        cons = {
            "role_or_skill": str(c.get("role_or_skill", "") or ""),
            "seniority": str(c.get("seniority", "") or ""),
            "test_types_include": [str(x).upper() for x in (c.get("test_types_include") or [])],
            "test_types_exclude": [str(x).upper() for x in (c.get("test_types_exclude") or [])],
            "duration_limit_min": c.get("duration_limit_min"),
            "remote_required": bool(c.get("remote_required", False)),
        }
        query = str(raw.get("search_query", "") or "").strip() or user_text(msgs)
        return State(
            intent=intent,
            enough=bool(raw.get("enough_context_to_recommend", False)),
            cons=cons,
            query=query,
            compare=[str(x) for x in (raw.get("compare_targets") or [])],
            user_turns=users,
            asst_turns=assts,
            total=total,
        )

    @staticmethod
    def _fallback(msgs: list[dict], users: int) -> State:
        # Used when the LLM call fails: conservative, never crashes.
        last = next(
            (m.get("content", "") for m in reversed(msgs) if (m.get("role") or "") == "user"),
            "",
        )
        if Guard.is_injection(last):
            intent = "injection"
        elif Guard.is_offtopic(last):
            intent = "off_topic"
        elif users <= 1 and len(last.split()) < 5:
            intent = "clarify"
        else:
            intent = "recommend"
        return State(
            intent=intent,
            enough=(intent == "recommend"),
            query=user_text(msgs),
            user_turns=users,
        )
