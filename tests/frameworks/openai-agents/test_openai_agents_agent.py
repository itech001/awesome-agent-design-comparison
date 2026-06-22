from agents import Agent

from agent import build_agent, DEFAULT_MODEL, AGENT_INSTRUCTIONS
from output_models import AgentAnswer


def test_build_agent_returns_sdk_agent():
    a = build_agent()
    assert isinstance(a, Agent)


def test_agent_has_expected_model():
    a = build_agent()
    assert a.model == DEFAULT_MODEL


def test_agent_uses_structured_output_type():
    a = build_agent()
    assert a.output_type is AgentAnswer


def test_instructions_mention_exam_context():
    assert "senior secondary school entrance exam" in AGENT_INSTRUCTIONS.lower()
    assert "multiple choice" in AGENT_INSTRUCTIONS.lower()


def test_build_agent_accepts_override_model():
    a = build_agent(model="gpt-4o-mini")
    assert a.model == "gpt-4o-mini"
