from google.adk.agents import LlmAgent

from agent import build_agent, DEFAULT_MODEL, AGENT_INSTRUCTION
from output_models import AgentAnswer


def test_build_agent_returns_llm_agent():
    a = build_agent()
    assert isinstance(a, LlmAgent)


def test_agent_has_expected_model():
    a = build_agent()
    assert a.model == DEFAULT_MODEL


def test_agent_uses_structured_output_schema():
    a = build_agent()
    assert a.output_schema is AgentAnswer


def test_instructions_mention_exam_context():
    assert "Zhongkao" in AGENT_INSTRUCTION
    assert "multiple choice" in AGENT_INSTRUCTION.lower()


def test_build_agent_accepts_override_model():
    a = build_agent(model="gemini-1.5-flash")
    assert a.model == "gemini-1.5-flash"
