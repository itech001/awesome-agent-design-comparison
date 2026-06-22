"""Entrypoint: solve the dataset with the OpenAI Agents SDK and write result/results.json.

Usage (contract):
    python run.py --dataset ../../dataset/questions.json --output result/results.json

Idiomatic SDK: build one Agent, solve each question via Runner, write the shared
result schema. This file is run as a script (python run.py), so it uses flat
imports (output_models, agent, solver) resolved from this directory.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# When run as a script, ensure this directory is on sys.path for flat imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import click
from agents import Agent
from dotenv import load_dotenv

from agent import DEFAULT_MODEL, build_agent
from output_models import AgentAnswer  # noqa: F401  (re-exported for convenience)
from solver import RunnerFn, solve_one

load_dotenv()


def run(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    framework: str = "openai-agents",
    model: str = DEFAULT_MODEL,
    runner: RunnerFn | None = None,
) -> dict:
    """Solve every question in the dataset and write the result file.

    Returns the written result dict. `runner` is injectable for tests.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(dataset_path.read_text())
    # Build the agent once (production) when using the real runner. When a
    # runner is injected (tests), pass None — the fake runner ignores the
    # agent argument, and solve_one's lazy build_agent() is cheap (no API call).
    if runner is None:
        agent = build_agent(model=model)
    else:
        agent = None

    results = []
    for q in questions:
        qi_id = q["id"]
        qi_subject = q["subject"]
        qi_type = q["type"]
        try:
            answer, latency_ms = solve_one(q, runner=runner, agent=agent)
            results.append({
                "question_id": qi_id,
                "subject": qi_subject,
                "type": qi_type,
                "response": answer.response,
                "reasoning": answer.reasoning,
                "latency_ms": latency_ms,
                "raw": {"agent": "ExamSolver"},
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
        "config": {"temperature": 0.0},
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


@click.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def cli(dataset_path: str, output_path: str, model: str) -> None:
    """Solve the dataset and write OUTPUT in the shared result schema."""
    payload = run(dataset_path=dataset_path, output_path=output_path, model=model)
    click.echo(f"{payload['framework']}: wrote {len(payload['results'])} answers to {output_path}")


if __name__ == "__main__":
    cli()
