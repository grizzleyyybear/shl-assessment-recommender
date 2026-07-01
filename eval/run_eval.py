"""Run the full evaluation suite and write eval/eval_report.md.

This is our evidence for "what didn't work and how we measured improvement" in the approach doc.
Re-run after every meaningful change.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.catalog import load_catalog  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from eval import behavior_probes, replay_harness  # noqa: E402

REPORT = Path(__file__).parent / "eval_report.md"


def main() -> None:
    catalog = load_catalog()
    retriever = Retriever(catalog)

    recall = replay_harness.run(catalog, retriever)
    pool = replay_harness.pool_recall(catalog, retriever)
    ground = replay_harness.groundedness(catalog, retriever)
    probes = behavior_probes.run(catalog, retriever)

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append(f"_Catalog items: {len(catalog)}_")
    lines.append("")
    lines.append("The suite measures the four things the brief asks for:")
    lines.append("")
    lines.append("| Measure | Metric | Score |")
    lines.append("|---------|--------|-------|")
    lines.append(f"| Retrieval quality | Mean candidate-pool recall (ceiling) | "
                 f"**{pool['mean_pool_recall']:.3f}** |")
    lines.append(f"| Recommendation relevance | Mean Recall@10 vs reference shortlists | "
                 f"**{recall['mean_recall@10']:.3f}** |")
    lines.append(f"| Groundedness | Returned URLs resolving to catalog entries | "
                 f"**{ground['groundedness']:.3f}** ({ground['grounded']}/{ground['recommendations']}) |")
    lines.append(f"| Response accuracy | Behavior-probe pass rate | "
                 f"**{probes['pass_rate']:.3f}** |")
    lines.append("")
    lines.append("## Recommendation relevance — Recall@10 (deterministic replay of provided traces)")
    lines.append("")
    lines.append("| Trace | Recall@10 | Hits | Expected | Returned |")
    lines.append("|-------|-----------|------|----------|----------|")
    for row in recall["rows"]:
        lines.append(
            f"| {row['trace']} | {row['recall@10']:.3f} | {row['hits']} | "
            f"{row['expected']} | {row['returned']} |"
        )
    lines.append(f"| **Mean** | **{recall['mean_recall@10']:.3f}** | | | |")
    lines.append("")
    lines.append("## Retrieval quality — candidate-pool recall (the ceiling the selector can't exceed)")
    lines.append("")
    lines.append("| Trace | Pool recall | Found | Expected |")
    lines.append("|-------|-------------|-------|----------|")
    for row in pool["rows"]:
        lines.append(
            f"| {row['trace']} | {row['pool_recall']:.3f} | {row['found']} | {row['expected']} |"
        )
    lines.append(f"| **Mean** | **{pool['mean_pool_recall']:.3f}** | | |")
    lines.append("")
    lines.append("## Groundedness")
    lines.append("")
    lines.append(
        f"{ground['grounded']}/{ground['recommendations']} returned recommendations resolve to a "
        f"real catalog entry (**{ground['groundedness']:.3f}**). URLs are projected from the catalog "
        f"by ID, so a hallucinated link is structurally impossible."
    )
    lines.append("")
    lines.append("## Response accuracy — behavior probes")
    lines.append("")
    lines.append("| Probe | Result | Assertion |")
    lines.append("|-------|--------|-----------|")
    for row in probes["results"]:
        status = "PASS" if row["pass"] else "FAIL"
        lines.append(f"| {row['probe']} | {status} | {row['desc']} |")
    lines.append(f"| **Pass rate** | **{probes['pass_rate']:.3f}** | |")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
