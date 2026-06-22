"""Build the OpenAI Agents SDK Agent.

Idiomatic SDK style: a single Agent with instructions + model + structured
output_type. No handoffs, no tools (the exam questions don't benefit from
external tools, and combining tools with structured output can suppress tool
calls — see SDK known issues). This keeps the implementation minimal and
showcases the SDK's "agent = instructions + model + output_type" core.
"""
from __future__ import annotations

from agents import Agent

from output_models import AgentAnswer

DEFAULT_MODEL = "gpt-4o"

AGENT_INSTRUCTIONS = """\
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


def build_agent(*, model: str = DEFAULT_MODEL) -> Agent:
    """Construct the solver Agent."""
    return Agent(
        name="ExamSolver",
        instructions=AGENT_INSTRUCTIONS,
        model=model,
        output_type=AgentAnswer,
    )
