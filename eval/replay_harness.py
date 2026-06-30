"""Replay harness: run each trace's user turns against the agent and compute Recall@K.

Deterministic replay (default): feeds the persona's recorded user turns in order. This is fast and
great for regression, but note its limitation — if our agent asks a clarifying question the original
trace didn't, the next scripted user turn may not answer it. We mitigate by always committing a
shortlist within the turn budget, so a final set is still produced for scoring.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent import Agent  # noqa: E402
from app.catalog import load_catalog  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from eval.expected import Trace, load_traces  # noqa: E402

THROTTLE_SECONDS = float(os.getenv("EVAL_THROTTLE", "1.2"))


def _norm(url: str) -> str:
    return url.rstrip("/").rstrip(".")


def replay(trace: Trace, ag: Agent, max_turns: int = 8) -> list[dict]:
    messages: list[dict] = []
    final_recs: list[dict] = []
    turn_count = 0
    for user_turn in trace.user_turns:
        if turn_count >= max_turns:
            break
        messages.append({"role": "user", "content": user_turn})
        turn_count += 1
        resp = ag.handle(messages)
        messages.append({"role": "assistant", "content": resp["reply"]})
        turn_count += 1
        if resp["recommendations"]:
            final_recs = resp["recommendations"]
        if resp["end_of_conversation"]:
            break
        time.sleep(THROTTLE_SECONDS)  # respect free-tier RPM during bursty local eval
    return final_recs


def recall_at_k(predicted_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    if not expected_urls:
        return 1.0
    pred = {_norm(u) for u in predicted_urls[:k]}
    exp = {_norm(u) for u in expected_urls}
    return len(pred & exp) / len(exp)


def run(catalog=None, retriever=None) -> dict:
    cat = catalog or load_catalog()
    ret = retriever or Retriever(cat)
    ag = Agent(cat, ret)
    traces = load_traces()
    rows = []
    total = 0.0
    for t in traces:
        recs = replay(t, ag)
        pred_urls = [r["url"] for r in recs]
        r10 = recall_at_k(pred_urls, t.expected_urls, k=10)
        total += r10
        hits = len({_norm(u) for u in pred_urls} & {_norm(u) for u in t.expected_urls})
        rows.append({
            "trace": t.name,
            "recall@10": round(r10, 3),
            "hits": hits,
            "expected": len(t.expected_urls),
            "returned": len(recs),
        })
    mean = total / len(traces) if traces else 0.0
    return {"rows": rows, "mean_recall@10": round(mean, 3)}


if __name__ == "__main__":
    result = run()
    for row in result["rows"]:
        print(
            f"{row['trace']:6} recall@10={row['recall@10']:.3f} "
            f"hits={row['hits']}/{row['expected']} returned={row['returned']}"
        )
    print(f"\nMean Recall@10: {result['mean_recall@10']:.3f}")
