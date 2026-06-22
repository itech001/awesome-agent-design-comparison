"""Solve one question via the two-loop design (docs/agents-design.md).

Inner loop: Resolver produces an answer, Validator independently checks it.
The loop repeats (feeding the Validator's feedback back to the Resolver) until
the Validator accepts or MAX_ATTEMPTS is reached.

Runners are injectable so tests stay deterministic and free of API calls. Each
runner takes (agent, prompt) and returns the agent's typed output object.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import Agent, Runner

from output_models import (
    AgentAnswer,
    MAX_ATTEMPTS,
    ResolverOutput,
    ValidatorVerdict,
    QuestionStatus,
    normalize_response,
)

ResolverRunner = Callable[[Agent, str], ResolverOutput]
ValidatorRunner = Callable[[Agent, str], ValidatorVerdict]


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


# Keep the old name as an alias so existing tests that import build_prompt still work.
build_prompt = build_question_prompt


def _default_resolver_runner(agent: Agent, prompt: str) -> ResolverOutput:
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def _default_validator_runner(agent: Agent, prompt: str) -> ValidatorVerdict:
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def _build_resolver_prompt(q: dict[str, Any], feedback: str = "") -> str:
    base = build_question_prompt(q)
    if feedback:
        return (
            f"{base}\n\n"
            "A previous attempt at this question was rejected. Revise your answer.\n"
            f"Examiner feedback: {feedback}\n\n"
            "Remember: respond with a single capital letter for multiple choice, "
            "or the complete answer for short answer."
        )
    return (
        f"{base}\n\n"
        "Remember: respond with a single capital letter for multiple choice, "
        "or the complete answer for short answer."
    )


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
    resolver: Agent,
    validator: Agent,
    resolver_runner: ResolverRunner,
    validator_runner: ValidatorRunner,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[AgentAnswer, QuestionStatus]:
    """Run resolve->validate until accepted or max_attempts. Returns (final answer, status)."""
    qi = QuestionInput.from_question(q)
    feedback = ""
    last_answer: AgentAnswer | None = None
    validator_answer: str | None = None
    accepted = False
    attempts = 0

    for attempts in range(1, max_attempts + 1):
        resolved: ResolverOutput = resolver_runner(
            resolver, _build_resolver_prompt(q, feedback)
        )
        last_answer = AgentAnswer(
            response=normalize_response(resolved.response, is_mc=qi.is_mc),
            reasoning=resolved.reasoning,
        )
        verdict: ValidatorVerdict = validator_runner(
            validator, _build_validator_prompt(q, resolved.response)
        )
        validator_answer = normalize_response(verdict.checked_answer, is_mc=qi.is_mc)
        if verdict.accepted:
            accepted = True
            break
        feedback = verdict.feedback

    # last_answer is guaranteed set because max_attempts >= 1.
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
    runner: Callable[[Agent, str], Any] | None = None,
    agent: Agent | None = None,
    resolver_agent: Agent | None = None,
    validator_agent: Agent | None = None,
    resolver_runner: ResolverRunner | None = None,
    validator_runner: ValidatorRunner | None = None,
    enable_validator: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (final AgentAnswer, latency_ms).

    Multi-agent path (default, enable_validator=True): runs the two-loop design
    with Resolver + Validator. The QuestionStatus is attached to the returned
    AgentAnswer via .__status__ so the caller (run.py) can record attempts /
    accepted / validator_answer in `raw`.

    Single-agent path (enable_validator=False, or the legacy `runner`/`agent`
    kwargs used by old tests): runs the Resolver only, once.
    """
    qi = QuestionInput.from_question(q)

    # Legacy single-agent path: a bare `runner` returns AgentAnswer and is the
    # entire solve. Used by the old Phase 2/3 test fakes.
    if runner is not None and resolver_runner is None and validator_runner is None:
        ag = agent
        if ag is None:
            from agent import build_resolver
            ag = build_resolver()
        prompt = _build_resolver_prompt(q)
        start = time.perf_counter()
        raw = runner(ag, prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = AgentAnswer(
            response=normalize_response(raw.response, is_mc=qi.is_mc),
            reasoning=raw.reasoning,
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
    r_run = resolver_runner or _default_resolver_runner
    v_run = validator_runner or _default_validator_runner

    if not enable_validator:
        start = time.perf_counter()
        raw = r_run(r_agent, _build_resolver_prompt(q))
        latency_ms = int((time.perf_counter() - start) * 1000)
        answer = AgentAnswer(
            response=normalize_response(raw.response, is_mc=qi.is_mc),
            reasoning=raw.reasoning,
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
