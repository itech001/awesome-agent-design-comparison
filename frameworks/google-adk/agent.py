"""Build the Google ADK LlmAgent.

Idiomatic ADK style: an LlmAgent with name + model + instruction + output_schema.
ADK enforces that output_schema and tools are mutually exclusive — since exam
questions need structured output and don't benefit from tools, we use
output_schema (no tools). This showcases ADK's code-first, schema-driven core.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from output_models import AgentAnswer

DEFAULT_MODEL = "gemini-2.5-flash"

AGENT_INSTRUCTION = """\
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


def build_agent(*, model: str = DEFAULT_MODEL) -> LlmAgent:
    """Construct the solver LlmAgent."""
    return LlmAgent(
        name="ExamSolver",
        model=model,
        instruction=AGENT_INSTRUCTION,
        output_schema=AgentAnswer,
    )
