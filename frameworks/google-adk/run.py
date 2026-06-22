"""Entrypoint: solve the dataset with Google ADK and write result/results.json.

Usage (contract):
    python run.py --dataset ../../dataset/questions.json --output result/results.json

Implements the two-loop multi-agent design from docs/agents-design.md:
  - outer loop: walk the dataset (every question addressed);
  - inner loop: Resolver -> Validator until accepted or MAX_ATTEMPTS.

Run as a script; flat imports from this directory.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import click
from dotenv import load_dotenv

from agent import DEFAULT_MODEL, build_resolver, build_validator
from solver import RunnerFn, solve_one, get_status

load_dotenv()


def run(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    framework: str = "google-adk",
    model: str = DEFAULT_MODEL,
    # Legacy single-runner hook: when set, runs the single-agent path (old tests).
    runner: RunnerFn | None = None,
    # Two-loop hooks: when set, runs the Resolver -> Validator design.
    resolver_runner: RunnerFn | None = None,
    validator_runner: RunnerFn | None = None,
    enable_validator: bool = True,
) -> dict:
    """Solve every question in the dataset and write the result file."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(dataset_path.read_text())

    resolver_agent = None
    validator_agent = None
    if runner is None and resolver_runner is None:
        resolver_agent = build_resolver(model=model)
        if enable_validator:
            validator_agent = build_validator(model=model)

    results = []
    for q in questions:
        qi_id, qi_subject, qi_type = q["id"], q["subject"], q["type"]
        try:
            answer, latency_ms = solve_one(
                q,
                runner=runner,
                resolver_agent=resolver_agent,
                validator_agent=validator_agent,
                resolver_runner=resolver_runner,
                validator_runner=validator_runner,
                enable_validator=enable_validator,
            )
            status = get_status(answer)
            raw = {"agent": "ExamSolver"}
            if status is not None:
                raw.update({
                    "attempts": status.attempts,
                    "accepted": status.accepted,
                    "validator_answer": status.validator_answer,
                })
            results.append({
                "question_id": qi_id,
                "subject": qi_subject,
                "type": qi_type,
                "response": answer.response,
                "reasoning": answer.reasoning,
                "latency_ms": latency_ms,
                "raw": raw,
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "question_id": qi_id,
                "subject": qi_subject,
                "type": qi_type,
                "response": "",
                "reasoning": "",
                "latency_ms": 0,
                "raw": {"agent": "ExamSolver", "error": str(exc)},
            })

    payload = {
        "framework": framework,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {"temperature": 0.0, "enable_validator": enable_validator},
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


@click.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--no-validator", is_flag=True, default=False,
    help="Disable the Validator agent (single-agent baseline).",
)
def cli(dataset_path: str, output_path: str, model: str, no_validator: bool) -> None:
    """Solve the dataset and write OUTPUT in the shared result schema."""
    payload = run(
        dataset_path=dataset_path,
        output_path=output_path,
        model=model,
        enable_validator=not no_validator,
    )
    click.echo(f"{payload['framework']}: wrote {len(payload['results'])} answers to {output_path}")


if __name__ == "__main__":
    cli()
