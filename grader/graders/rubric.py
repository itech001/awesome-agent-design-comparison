"""Deterministic rubric grader for short-answer questions.

For each criterion, checks whether the response contains matching keywords.
Reproducible: no LLM, no randomness.
"""
from __future__ import annotations

import re

from grader.graders.base import Grader
from grader.models import AnswerResult, CriterionScore, Question, Score

_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "is", "are",
    "for", "with", "on", "at", "by", "it", "this", "that", "its",
    "be", "as", "from", "side", "other", "one", "correct", "final",
    "answer", "mentions", "states", "uses", "value", "unit",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    # keep alphanumerics and a few math symbols
    text = re.sub(r"[^a-z0-9+\-^/=(). ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(criterion: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", criterion.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _notes_keywords(notes: str) -> list[str]:
    """Extract synonym phrases from notes like 'accept: five; sqrt(25)'."""
    if not notes:
        return []
    m = re.search(r"accept\s*:\s*(.+)", notes, re.IGNORECASE)
    if not m:
        return []
    body = m.group(1)
    return [p.strip() for p in body.split(";") if p.strip()]


class RubricGrader(Grader):
    name = "rubric"

    def score(self, question: Question, answer: AnswerResult) -> Score:
        response_norm = _normalize(answer.response)
        notes_synonyms = _notes_keywords(question.notes)

        per: list[CriterionScore] = []
        earned = 0
        max_points = question.scoring.max_points
        for crit in question.scoring.rubric or []:
            keywords = _keywords(crit.criterion)
            if keywords:
                matched = any(k in response_norm for k in keywords)
            else:
                # criterion is built only from common words; fall back to
                # matching the whole normalized criterion phrase as a substring
                phrase = _normalize(crit.criterion)
                matched = bool(phrase) and phrase in response_norm
            # also check notes synonyms (whole-phrase match on normalized text)
            if not matched:
                for syn in notes_synonyms:
                    if _normalize(syn) and _normalize(syn) in response_norm:
                        matched = True
                        break
            if matched:
                earned += crit.points
            per.append(CriterionScore(
                criterion=crit.criterion,
                earned=crit.points if matched else 0,
                available=crit.points,
            ))

        correct = earned == max_points and max_points > 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=per,
            grader=self.name,
        )
