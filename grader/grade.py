"""CLI: grade a single framework result file.

Usage:
    python -m grader.grade --result <result.json> --dataset <questions.json> [--judge llm]
"""
from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from grader.loaders import load_questions_by_id, load_result_file
from grader.report import render_per_framework
from grader.score import aggregate, score_result_file

load_dotenv()


@click.command()
@click.option("--result", "result_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--judge", type=click.Choice(["rubric", "llm"]), default="rubric",
              help="rubric (default, deterministic) or llm (LLM-as-judge)")
@click.option("--report-dir", default=None, type=click.Path(),
              help="Optional dir to write a markdown report")
def cli(result_path: str, dataset_path: str, judge: str, report_dir: str | None) -> None:
    rf = load_result_file(result_path)
    scored = score_result_file(rf, dataset_path=dataset_path, judge=judge)
    type_by_qid = {qid: q.type for qid, q in load_questions_by_id(dataset_path).items()}
    agg = aggregate(scored, type_by_qid=type_by_qid)
    click.echo(f"{rf.framework}: overall={agg['overall']:.1%} ({agg['count']} questions)")
    if report_dir:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = render_per_framework(rf.framework, rf.model, agg, scored)
        out = out_dir / f"{rf.framework}.md"
        out.write_text(md)
        click.echo(f"report -> {out}")


if __name__ == "__main__":
    cli()
