"""Build the Google ADK LlmAgents for the two-loop design.

The design (docs/agents-design.md) uses two agents in the inner loop:
  - Resolver: produces {response, reasoning} for one question.
  - Validator: independently checks the answer and returns {accepted, feedback,
    checked_answer}.

Both use ADK's structured output_schema. No tools (ADK enforces that
output_schema and tools are mutually exclusive).
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from output_models import ResolverOutput, ValidatorVerdict

DEFAULT_MODEL = "gemini-2.5-flash"

RESOLVER_INSTRUCTION = """\
You are a top student taking the senior secondary school entrance exam.
Answer each question correctly and concisely.

For multiple choice questions, `response` MUST be a single capital letter
(A, B, C, or D) corresponding to the correct option.

For short answer questions, `response` MUST be the complete answer (a number
with units, a short phrase, a balanced equation, or a one-sentence
explanation) — exactly as a correct exam answer would be written.

Always include brief `reasoning` showing how you reached the answer.
Respond ONLY with the JSON schema fields (response and reasoning). Do not add
extraneous commentary.
"""

VALIDATOR_INSTRUCTION = """\
You are an independent examiner verifying a student's answer on the senior
secondary school entrance exam. Solve the question yourself first, from first
principles, WITHOUT trusting the student's answer.

Then compare your own answer to the student's. Set `accepted` to true only if
the student's answer is correct. If it is wrong, set `accepted` to false and
explain the mistake concisely in `feedback`.

Always record your own independently-derived answer in `checked_answer`.
Respond ONLY with the JSON schema fields (accepted, feedback, checked_answer).
"""


def build_resolver(*, model: str = DEFAULT_MODEL) -> LlmAgent:
    """Construct the Resolver LlmAgent."""
    return LlmAgent(
        name="Resolver",
        model=model,
        instruction=RESOLVER_INSTRUCTION,
        output_schema=ResolverOutput,
    )


def build_validator(*, model: str = DEFAULT_MODEL) -> LlmAgent:
    """Construct the Validator LlmAgent."""
    return LlmAgent(
        name="Validator",
        model=model,
        instruction=VALIDATOR_INSTRUCTION,
        output_schema=ValidatorVerdict,
    )


# Backwards compatibility: Phase 2/3 referenced build_agent / AGENT_INSTRUCTION.
def build_agent(*, model: str = DEFAULT_MODEL) -> LlmAgent:
    """Deprecated alias for build_resolver (single-agent baseline)."""
    return build_resolver(model=model)


AGENT_INSTRUCTION = RESOLVER_INSTRUCTION
