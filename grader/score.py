"""Pick the right grader per question and aggregate scores."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from grader.graders.base import Grader
from grader.graders.exact import ExactGrader
from grader.graders.llm_judge import LLMJudgeGrader, OpenAIJudgeClient
from grader.graders.rubric import RubricGrader
from grader.loaders import load_questions_by_id
from grader.models import ResultFile, Score

Judge = Literal["rubric", "llm"]


def _select_grader(question, *, judge: Judge) -> Grader:
    if question.answer_type == "letter":
        return ExactGrader()
    # text / rubric
    if judge == "llm":
        return LLMJudgeGrader(client=OpenAIJudgeClient())
    return RubricGrader()


def score_result_file(
    result_file: ResultFile,
    *,
    dataset_path: str | Path,
    judge: Judge = "rubric",
) -> list[Score]:
    """Score every answer in a result file. Unknown question_ids are skipped."""
    by_id = load_questions_by_id(dataset_path)
    scored: list[Score] = []
    for ans in result_file.results:
        q = by_id.get(ans.question_id)
        if q is None:
            continue
        grader = _select_grader(q, judge=judge)
        scored.append(grader.score(q, ans))
    return scored


def aggregate(scored: list[Score]) -> dict:
    """Aggregate a list of per-question Scores into summary stats."""
    if not scored:
        return {"overall": 0.0, "by_subject": {}, "by_type": {}, "count": 0}

    overall = sum(s.normalized for s in scored) / len(scored)

    def _subject(qid: str) -> str:
        return qid.rsplit("-", 1)[0]

    def _type(score: Score) -> str:
        # heuristic: letter-answer graders produce empty per_criterion -> mc
        return "multiple_choice" if not score.per_criterion else "short_answer"

    subj_sums: dict[str, list[float]] = defaultdict(list)
    type_sums: dict[str, list[float]] = defaultdict(list)
    for s in scored:
        subj_sums[_subject(s.question_id)].append(s.normalized)
        type_sums[_type(s)].append(s.normalized)

    return {
        "overall": overall,
        "by_subject": {k: sum(v) / len(v) for k, v in subj_sums.items()},
        "by_type": {k: sum(v) / len(v) for k, v in type_sums.items()},
        "count": len(scored),
    }
