"""LLM wrapper for the LangGraph nodes.

LangGraph is model-agnostic; we use langchain-openai's ChatOpenAI by default
(gpt-4o per the project spec). Node functions call invoke(prompt) -> str, so
tests can inject any object with a compatible .invoke.
"""
from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "gpt-4o"


class LLM(Protocol):
    """Anything with an .invoke(prompt) -> str method is an LLM here."""

    def invoke(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


def build_llm(*, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> LLM:
    """Construct the default ChatOpenAI LLM. Requires OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI  # local import: optional dependency
    return ChatOpenAI(model=model, temperature=temperature)
