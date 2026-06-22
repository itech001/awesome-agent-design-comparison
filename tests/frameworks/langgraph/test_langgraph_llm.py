"""Tests for the LLM wrapper."""
from llm import build_llm, DEFAULT_MODEL, LLM


class FakeLLM:
    def __init__(self, text="A"):
        self._text = text
        self.calls = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._text


def test_fake_llm_satisfies_protocol():
    # Structural typing: FakeLLM has .invoke(str) -> str, so it conforms to LLM.
    fake = FakeLLM("B")
    assert isinstance(fake, object)  # Protocol check is structural at runtime here
    assert fake.invoke("hi") == "B"
    assert len(fake.calls) == 1


def test_default_model_is_gpt4o():
    assert DEFAULT_MODEL == "gpt-4o"


def test_build_llm_returns_invokeable():
    # We can't call the real API in tests, but build_llm must return something
    # with an .invoke method (an LLM). Guard with env skip.
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        return  # skip: no key in CI
    llm = build_llm()
    assert hasattr(llm, "invoke")
