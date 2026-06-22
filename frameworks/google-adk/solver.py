"""Solve one question via the two-loop design (docs/agents-design.md).

Inner loop: Resolver produces an answer, Validator independently checks it.
The loop repeats (feeding the Validator's feedback back to the Resolver) until
the Validator accepts or MAX_ATTEMPTS is reached.

Runners are injectable so tests stay deterministic and free of API calls. Each
runner takes (agent, prompt) and returns the agent's final response TEXT (ADK
serializes output_schema to JSON text); the solver parses it back.
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

from output_models import (
    AgentAnswer,
    MAX_ATTEMPTS,
    QuestionStatus,
    normalize_response,
    parse_final_text,
    parse_validator_text,
)

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


def build_question_prompt(q: dict[str, Any]) -> str:
    """Render the question text for both Resolver and Validator."""
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
    return "\n".join(lines)


# Alias for old tests that import build_prompt.
build_prompt = build_question_prompt


def _default_runner(agent: LlmAgent, prompt: str) -> str:
    """Run the agent via ADK's async Runner and return the final response text."""
    app_name = "exam_solver"
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


def _build_resolver_prompt(q: dict[str, Any], feedback: str = "") -> str:
    base = build_question_prompt(q)
    suffix = (
        "Remember: respond with a single capital letter for multiple choice, "
        "or the complete answer for short answer."
    )
    if feedback:
        return (
            f"{base}\n\n"
            "A previous attempt at this question was rejected. Revise your answer.\n"
            f"Examiner feedback: {feedback}\n\n"
            + suffix
        )
    return f"{base}\n\n{suffix}"


def _build_validator_prompt(q: dict[str, Any], student_answer: str) -> str:
    base = build_question_prompt(q)
    return (
        f"{base}\n\n"
        f"Student's answer to verify: {student_answer}\n\n"
        "Solve the question yourself, then decide whether the student's answer "
        "is correct. For multiple choice, `checked_answer` must be a single "
        "capital letter; for short answer, the complete correct answer."
    )


def _run_inner_loop(
    q: dict[str, Any],
    *,
    resolver: LlmAgent,
    validator: LlmAgent,
    resolver_runner: RunnerFn,
    validator_runner: RunnerFn,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[AgentAnswer, QuestionStatus]:
    """Run resolve->validate until accepted or max_attempts."""
    qi = QuestionInput.from_question(q)
    feedback = ""
    last_answer: AgentAnswer | None = None
    validator_answer: str | None = None
    accepted = False
    attempts = 0

    for attempts in range(1, max_attempts + 1):
        raw_text = resolver_runner(resolver, _build_resolver_prompt(q, feedback))
        resolved = parse_final_text(raw_text, is_mc=qi.is_mc)
        last_answer = AgentAnswer(
            response=normalize_response(resolved.response, is_mc=qi.is_mc),
            reasoning=resolved.reasoning,
        )
        v_text = validator_runner(
            validator, _build_validator_prompt(q, resolved.response)
        )
        verdict = parse_validator_text(v_text, is_mc=qi.is_mc)
        validator_answer = normalize_response(verdict.checked_answer, is_mc=qi.is_mc)
        if verdict.accepted:
            accepted = True
            break
        feedback = verdict.feedback

    assert last_answer is not None
    status = QuestionStatus(
        question_id=qi.id,
        final_response=last_answer.response,
        reasoning=last_answer.reasoning,
        accepted=accepted,
        attempts=attempts,
        validator_answer=validator_answer,
    )
    return last_answer, status


def solve_one(
    q: dict[str, Any],
    *,
    runner: RunnerFn | None = None,
    agent: LlmAgent | None = None,
    resolver_agent: LlmAgent | None = None,
    validator_agent: LlmAgent | None = None,
    resolver_runner: RunnerFn | None = None,
    validator_runner: RunnerFn | None = None,
    enable_validator: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (final AgentAnswer, latency_ms).

    Multi-agent path (default): runs the two-loop design with Resolver +
    Validator. The QuestionStatus is attached to the returned AgentAnswer via
    .__status__.

    Legacy single-runner path (runner set, no resolver_runner/validator_runner):
    runs the Resolver only, once — used by the Phase 2/3 test fakes.
    """
    qi = QuestionInput.from_question(q)

    # Legacy single-agent path.
    if runner is not None and resolver_runner is None and validator_runner is None:
        ag = agent
        if ag is None:
            from agent import build_resolver
            ag = build_resolver()
        prompt = _build_resolver_prompt(q)
        start = time.perf_counter()
        raw_text = runner(ag, prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)
        parsed = parse_final_text(raw_text, is_mc=qi.is_mc)
        answer = AgentAnswer(
            response=normalize_response(parsed.response, is_mc=qi.is_mc),
            reasoning=parsed.reasoning,
        )
        answer.__status__ = QuestionStatus(  # type: ignore[attr-defined]
            question_id=qi.id,
            final_response=answer.response,
            reasoning=answer.reasoning,
            accepted=True,
            attempts=1,
            validator_answer=None,
            latency_ms=latency_ms,
        )
        return answer, latency_ms

    # Multi-agent two-loop path.
    from agent import build_resolver, build_validator

    r_agent = resolver_agent or build_resolver()
    v_agent = validator_agent or build_validator()
    r_run = resolver_runner or _default_runner
    v_run = validator_runner or _default_runner

    if not enable_validator:
        start = time.perf_counter()
        raw_text = r_run(r_agent, _build_resolver_prompt(q))
        latency_ms = int((time.perf_counter() - start) * 1000)
        parsed = parse_final_text(raw_text, is_mc=qi.is_mc)
        answer = AgentAnswer(
            response=normalize_response(parsed.response, is_mc=qi.is_mc),
            reasoning=parsed.reasoning,
        )
        answer.__status__ = QuestionStatus(  # type: ignore[attr-defined]
            question_id=qi.id,
            final_response=answer.response,
            reasoning=answer.reasoning,
            accepted=True,
            attempts=1,
            validator_answer=None,
            latency_ms=latency_ms,
        )
        return answer, latency_ms

    start = time.perf_counter()
    answer, status = _run_inner_loop(
        q,
        resolver=r_agent,
        validator=v_agent,
        resolver_runner=r_run,
        validator_runner=v_run,
        max_attempts=max_attempts,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    status.latency_ms = latency_ms
    answer.__status__ = status  # type: ignore[attr-defined]
    return answer, latency_ms


def get_status(answer: AgentAnswer) -> QuestionStatus | None:
    """Read the QuestionStatus attached by solve_one, if any."""
    return getattr(answer, "__status__", None)
