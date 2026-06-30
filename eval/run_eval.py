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
    probes = behavior_probes.run(catalog, retriever)

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append(f"_Catalog items: {len(catalog)}_")
    lines.append("")
    lines.append("## Recall@10 (deterministic replay of provided traces)")
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
    lines.append("## Behavior probes")
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
