"""CLI: grade every frameworks/*/result/results.json and write a comparison report.

Usage:
    python -m grader.grade_all [--judge llm] [--report-dir reports]
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

from grader.loaders import load_result_file
from grader.report import render_comparison, render_per_framework
from grader.score import aggregate, score_result_file

load_dotenv()

_REPO = Path(__file__).resolve().parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_FRAMEWORKS = _REPO / "frameworks"


def _discover_results() -> list[Path]:
    if not _FRAMEWORKS.exists():
        return []
    return sorted(_FRAMEWORKS.glob("*/result/results.json"))


@click.command()
@click.option("--judge", type=click.Choice(["rubric", "llm"]), default="rubric")
@click.option("--report-dir", default="reports", type=click.Path())
def cli(judge: str, report_dir: str) -> None:
    results = _discover_results()
    if not results:
        click.echo("No frameworks/*/result/results.json found.")
        raise SystemExit(0)

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summaries = []
    for p in results:
        rf = load_result_file(p)
        scored = score_result_file(rf, dataset_path=_DATASET, judge=judge)
        agg = aggregate(scored)
        summaries.append({
            "framework": rf.framework,
            "model": rf.model,
            **agg,
        })
        md = render_per_framework(rf.framework, rf.model, agg, scored)
        (out_dir / f"{rf.framework}-{ts}.md").write_text(md)
        click.echo(f"{rf.framework}: {agg['overall']:.1%}")

    comparison = render_comparison(summaries)
    comp_path = out_dir / f"comparison-{ts}.md"
    comp_path.write_text(comparison)
    click.echo(f"comparison -> {comp_path}")


if __name__ == "__main__":
    cli()
