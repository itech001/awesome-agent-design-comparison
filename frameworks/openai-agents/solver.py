"""Solve one question with the OpenAI Agents SDK.

The runner is injectable so tests stay deterministic and free of API calls.
By default it uses `agents.Runner.run_sync`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import Agent, Runner

from output_models import AgentAnswer, normalize_response

RunnerFn = Callable[[Agent, str], AgentAnswer]
"""A runner takes (agent, prompt) and returns the agent's AgentAnswer.

The real Runner.run_sync returns a RunResult whose .final_output is the
AgentAnswer; we wrap it (see _default_runner) so this signature holds.
"""


@dataclass
class QuestionInput:
    id: str
    subject: str
    type: str
    question: str
    answer_type: str
    options: list[str] = field(default_factory=list)

    @property
    def is_mc(self) -> bool:
        return self.type == "multiple_choice"

    @classmethod
    def from_question(cls, q: dict[str, Any]) -> "QuestionInput":
        return cls(
            id=q["id"],
            subject=q["subject"],
            type=q["type"],
            question=q["question"],
            answer_type=q["answer_type"],
            options=q.get("options", []),
        )


def build_prompt(q: dict[str, Any]) -> str:
    """Render a dataset question into the prompt string for the agent."""
    qi = QuestionInput.from_question(q)
    kind = "multiple choice" if qi.is_mc else "short answer"
    lines = [
        f"Subject: {qi.subject}",
        f"Question type: {kind}",
        "",
        qi.question,
    ]
    if qi.is_mc and qi.options:
        lines += ["", "Options:"] + [f"  {o}" for o in qi.options]
    lines += [
        "",
        "Remember: respond with a single capital letter for multiple choice, "
        "or the complete answer for short answer.",
    ]
    return "\n".join(lines)


def _default_runner(agent: Agent, prompt: str) -> AgentAnswer:
    """Wrap Runner.run_sync so it returns the AgentAnswer directly."""
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def solve_one(
    q: dict[str, Any],
    *,
    runner: RunnerFn | None = None,
    agent: Agent | None = None,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (normalized AgentAnswer, latency_ms).

    If `agent` is None, a default agent is built. If `runner` is None, the real
    Runner.run_sync is used (requires OPENAI_API_KEY).
    """
    run = runner or _default_runner
    ag = agent
    if ag is None:
        from agent import build_agent  # lazy import to avoid circular dep
        ag = build_agent()

    qi = QuestionInput.from_question(q)
    prompt = build_prompt(q)

    start = time.perf_counter()
    raw = run(ag, prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    normalized = AgentAnswer(
        response=normalize_response(raw.response, is_mc=qi.is_mc),
        reasoning=raw.reasoning,
    )
    return normalized, latency_ms
