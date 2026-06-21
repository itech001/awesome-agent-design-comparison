import pytest
from pydantic import ValidationError

from output_models import AgentAnswer, normalize_response, parse_final_text


def test_agent_answer_ok():
    a = AgentAnswer(response="A", reasoning="difference of squares")
    assert a.response == "A"


def test_agent_answer_requires_response():
    with pytest.raises(ValidationError):
        AgentAnswer(response="", reasoning="")


def test_normalize_mc_letter():
    assert normalize_response("A", is_mc=True) == "A"


def test_normalize_mc_strips_and_uppercases():
    assert normalize_response(" Option B. ", is_mc=True) == "B"


def test_normalize_mc_picks_standalone_letter():
    assert normalize_response("I think it is C because...", is_mc=True) == "C"


def test_normalize_short_answer_kept_verbatim():
    assert normalize_response("x = 12", is_mc=False) == "x = 12"


def test_parse_final_text_json():
    text = '{"response": "A", "reasoning": "diff of squares"}'
    a = parse_final_text(text, is_mc=True)
    assert a.response == "A"
    assert a.reasoning == "diff of squares"


def test_parse_final_text_plain_falls_back_to_response():
    a = parse_final_text("B", is_mc=True)
    assert a.response == "B"
    assert a.reasoning == ""


def test_parse_final_text_empty_raises():
    with pytest.raises(ValueError):
        parse_final_text("", is_mc=True)
