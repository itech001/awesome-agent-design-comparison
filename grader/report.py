"""Render per-framework and cross-framework markdown reports."""
from __future__ import annotations

from datetime import datetime, timezone

from grader.models import Score

_SUBJECTS = ["chinese", "math", "english", "physics", "chemistry", "biology", "history", "geography"]


def render_per_framework(framework: str, model: str, agg: dict, scored: list[Score]) -> str:
    overall = agg["overall"]
    lines = [
        f"# {framework} — Results",
        "",
        f"- **Model:** {model}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- **Questions scored:** {agg.get('count', 0)}",
        f"- **Overall (normalized):** {overall:.1%}",
        "",
        "## By subject",
        "",
        "| Subject | Score |",
        "|---|---|",
    ]
    for subj in _SUBJECTS:
        val = agg["by_subject"].get(subj)
        if val is not None:
            lines.append(f"| {subj} | {val:.1%} |")
    lines += ["", "## By question type", "", "| Type | Score |", "|---|---|"]
    for t, v in agg["by_type"].items():
        lines.append(f"| {t} | {v:.1%} |")
    lines += ["", "## Per-question detail", "", "| Question | Earned/Max | Correct | Grader |", "|---|---|---|---|"]
    for s in scored:
        lines.append(f"| {s.question_id} | {int(s.earned)}/{s.max_points} | {'✓' if s.correct else '✗'} | {s.grader} |")
    return "\n".join(lines) + "\n"


def render_comparison(summaries: list[dict]) -> str:
    lines = [
        "# Framework Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Overall",
        "",
        "| framework | model | overall | questions |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| {s['framework']} | {s.get('model', '?')} | {s['overall']:.1%} | {s.get('count', '?')} |")

    lines += ["", "## By subject", "", "| framework | " + " | ".join(_SUBJECTS) + " |",
              "|---|" + "---|" * len(_SUBJECTS)]
    for s in summaries:
        row = [s["framework"]]
        for subj in _SUBJECTS:
            v = s.get("by_subject", {}).get(subj)
            row.append(f"{v:.1%}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
