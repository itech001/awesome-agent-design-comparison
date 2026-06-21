"""Solve one question with Google ADK.

The runner is injectable so tests stay deterministic and free of API calls.
By default it uses ADK's async Runner.run_async, wrapped synchronously.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from output_models import AgentAnswer, normalize_response, parse_final_text

RunnerFn = Callable[[LlmAgent, str], str]
"""A runner takes (agent, prompt) and returns the final response text."""


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


def _default_runner(agent: LlmAgent, prompt: str) -> str:
    """Run the agent via ADK's async Runner and return the final response text."""
    app_name = "zhongkao_solver"
    user_id = "solver"

    async def _run() -> str:
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=app_name, user_id=user_id
        )
        runner = Runner(
            agent=agent, app_name=app_name, session_service=session_service
        )
        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                parts = event.content.parts if event.content else []
                final_text = "".join(
                    p.text for p in parts if getattr(p, "text", None)
                )
                break
        return final_text

    return asyncio.run(_run())


def solve_one(
    q: dict[str, Any],
    *,
    runner: RunnerFn | None = None,
    agent: LlmAgent | None = None,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (normalized AgentAnswer, latency_ms)."""
    run = runner or _default_runner
    ag = agent
    if ag is None:
        from agent import build_agent  # lazy import to avoid circular dep
        ag = build_agent()

    qi = QuestionInput.from_question(q)
    prompt = build_prompt(q)

    start = time.perf_counter()
    final_text = run(ag, prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    parsed = parse_final_text(final_text, is_mc=qi.is_mc)
    normalized = AgentAnswer(
        response=normalize_response(parsed.response, is_mc=qi.is_mc),
        reasoning=parsed.reasoning,
    )
    return normalized, latency_ms
