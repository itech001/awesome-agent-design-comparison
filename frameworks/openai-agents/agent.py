"""Build the OpenAI Agents SDK agents for the two-loop design.

The design (docs/agents-design.md) uses two agents in the inner loop:
  - Resolver: produces {response, reasoning} for one question.
  - Validator: independently checks the answer and returns {accepted, feedback,
    checked_answer}.

Both use structured output via `output_type`. No tools (exam questions don't
benefit, and combining tools with structured output can suppress tool calls).
"""
from __future__ import annotations

from agents import Agent

from output_models import ResolverOutput, ValidatorVerdict

DEFAULT_MODEL = "gpt-4o"

RESOLVER_INSTRUCTIONS = """\
You are a top student taking the senior secondary school entrance exam.
Answer each question correctly and concisely.

For multiple choice questions, your `response` MUST be a single capital letter
(A, B, C, or D) corresponding to the correct option.

For short answer questions, your `response` MUST be the complete answer (e.g.
a number with units, a short phrase, a balanced equation, or a one-sentence
explanation) — exactly as a correct exam answer would be written.

Always include brief `reasoning` showing how you reached the answer.
Be precise. Do not add extraneous commentary in `response`.
"""

VALIDATOR_INSTRUCTIONS = """\
You are an independent examiner verifying a student's answer on the senior
secondary school entrance exam. Solve the question yourself first, from first
principles, WITHOUT trusting the student's answer.

Then compare your own answer to the student's. Set `accepted` to true only if
the student's answer is correct. If it is wrong, set `accepted` to false and
explain the mistake concisely in `feedback` (so the student can revise).

Always record your own independently-derived answer in `checked_answer`.
"""


def build_resolver(*, model: str = DEFAULT_MODEL) -> Agent:
    """Construct the Resolver agent."""
    return Agent(
        name="Resolver",
        instructions=RESOLVER_INSTRUCTIONS,
        model=model,
        output_type=ResolverOutput,
    )


def build_validator(*, model: str = DEFAULT_MODEL) -> Agent:
    """Construct the Validator agent."""
    return Agent(
        name="Validator",
        instructions=VALIDATOR_INSTRUCTIONS,
        model=model,
        output_type=ValidatorVerdict,
    )


# Backwards compatibility: Phase 2/3 tests referenced build_agent / AGENT_INSTRUCTIONS.
# Keep a single resolver alias for any callers that haven't migrated.
def build_agent(*, model: str = DEFAULT_MODEL) -> Agent:
    """Deprecated alias for build_resolver (single-agent baseline)."""
    return build_resolver(model=model)


AGENT_INSTRUCTIONS = RESOLVER_INSTRUCTIONS
