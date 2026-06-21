import pytest
from pydantic import ValidationError

from output_models import AgentAnswer, normalize_response


def test_agent_answer_ok():
    a = AgentAnswer(response="A", reasoning="difference of squares")
    assert a.response == "A"
    assert a.reasoning == "difference of squares"


def test_agent_answer_requires_fields():
    with pytest.raises(ValidationError):
        AgentAnswer(response="", reasoning="")


def test_normalize_mc_letter():
    assert normalize_response("A", is_mc=True) == "A"


def test_normalize_mc_strips_and_uppercases():
    assert normalize_response(" Option B. ", is_mc=True) == "B"


def test_normalize_mc_picks_first_letter():
    assert normalize_response("I think it is C because...", is_mc=True) == "C"


def test_normalize_short_answer_kept_verbatim():
    assert normalize_response("x = 12", is_mc=False) == "x = 12"


def test_normalize_mc_empty_falls_back_to_unknown():
    assert normalize_response("", is_mc=True) == ""
