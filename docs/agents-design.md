# Agents Design

This document describes the multi-agent architecture used across the framework
implementations in this project. It is the reference design every framework
folder (`frameworks/<name>/`) aims to realize, each in its own idiomatic style.

> **Scope note.** Phases 2 and 3 shipped a deliberately minimal baseline: a
> single solver agent per framework, so the shared contract and grader could be
> validated first. The design below is the **target architecture**. The
> remaining frameworks (LangGraph, CrewAI) and an upgrade pass for the existing
> frameworks will implement it.

## Goals

- **Correctness** — every question gets a correct, validated answer.
- **Verifiability** — no answer is trusted without an independent check.
- **Coverage** — the orchestrator guarantees the full dataset is addressed.
- **Comparability** — every framework realizes the same agent roles, so the
  benchmark measures framework design, not agent-design differences.

## Top-level architecture

```
                       ┌─────────────────────────┐
                       │   Orchestrator Agent    │  (root)
                       │   owns the outer loop   │
                       └────────────┬─────────────┘
                                    │  drives the outer loop:
                                    │  "are all questions addressed?"
                  ┌─────────────────┴───────────────────┐
                  ▼                                     ▼
        ┌──────────────────┐                  ┌──────────────────┐
        │  Resolver Agent  │  ─── answer ───▶ │ Validator Agent  │
        │  (inner loop)    │ ◀── verdict/ ─── │  (inner loop)    │
        │  solves question │     revision     │  checks answer   │
        └──────────────────┘                  └──────────────────┘
                  │
                  ▼  once the validator accepts
        ┌──────────────────┐
        │  Summary Report  │  (final step, root)
        │  Agent           │
        └──────────────────┘
```

Five agent roles in total:

| Role | Loop | Responsibility |
|---|---|---|
| **Orchestrator** (root) | outer | Drives overall progress; decides when all questions are addressed; hands off to the Summary Agent. |
| **Resolver** | inner | Produces an answer for one question. |
| **Validator** | inner | Independently checks the Resolver's answer; accepts or requests revision. |
| **Summary Report** | terminal (root) | After the outer loop completes, produces the final cross-framework report from the validated answers. |
| *(Phase 6)* **Report writer** | terminal | Writes `result/results.json` and the comparison markdown. |

## The two loops

### Inner loop — resolve, then validate (per question)

For each question, the Resolver and Validator iterate until the Validator
accepts or a maximum number of attempts is reached.

```
do {
    answer  = Resolver.solve(question, previous_feedback?)
    verdict = Validator.check(question, answer)   // ACCEPT | REVISE(reason)
    if verdict is REVISE:
        previous_feedback = verdict.reason
} while not accepted and attempts < MAX_ATTEMPTS
```

- **Resolver** — receives the question and (on a second attempt) the Validator's
  feedback. Returns `{response, reasoning}`.
- **Validator** — independent of the Resolver. It re-derives or checks the answer
  against the question and the Resolver's `reasoning`, then returns either
  `ACCEPT` or `REVISE(reason)`. It does **not** see the dataset's reference
  answer — it validates from first principles, exactly like a human examiner.
- **MAX_ATTEMPTS** bounds the loop (default 3) so a stubborn disagreement can't
  run forever. On exhaustion, the last Resolver answer is recorded with a flag.

> Why two agents rather than one self-checking agent? Separation of concerns and
> independence. The Resolver commits to an answer; the Validator attacks it. This
> catches arithmetic slips, misread questions, and over-confident reasoning — the
> classic failure modes on exam questions. A single self-checking agent tends to
> rubber-stamp its own answer.

### Outer loop — ensure full coverage

The Orchestrator walks the question set and tracks which questions have been
resolved-and-validated. After each inner-loop completion it asks: *"are all
questions addressed?"* If no, it moves to the next unanswered question. If yes,
it hands control to the Summary Report Agent.

```
addressed = {}
while not all(addressed.values()):
    q = next_unanswered_question()
    answered = run_inner_loop(q)
    addressed[q.id] = answered
run_summary_report(addressed)
```

The outer loop guarantees **coverage** — a core benchmark requirement. No
question can be silently dropped, because the Orchestrator only finishes when the
addressed set equals the full dataset.

## Handoff contract between agents

Agents communicate through typed messages (one Pydantic model each), so the
handoff is unambiguous and testable without any framework-specific ceremony.

```python
class ResolverOutput(BaseModel):
    response: str          # single letter (MC) or full text (short answer)
    reasoning: str

class ValidatorVerdict(BaseModel):
    accepted: bool
    feedback: str = ""     # empty when accepted; the revision note otherwise
    checked_answer: str    # the Validator's own independently-derived answer

class QuestionStatus(BaseModel):
    question_id: str
    final_response: str
    accepted: bool
    attempts: int
    validator_answer: str | None
```

The framework implementations map `QuestionStatus` onto the shared
`contract/result.schema.json` `answerResult`:

| Contract field | Source |
|---|---|
| `response` | `QuestionStatus.final_response` |
| `reasoning` | last `ResolverOutput.reasoning` |
| `raw.attempts` | `QuestionStatus.attempts` |
| `raw.validator_answer` | `QuestionStatus.validator_answer` |
| `raw.accepted` | `QuestionStatus.accepted` |

The `raw` object carries the multi-agent detail without breaking comparability
with single-agent baselines.

## How each framework realizes this

Each `frameworks/<name>/` folder implements the same five roles in its own
idiom. The mapping:

| Framework | Orchestrator | Resolver | Validator | Summary |
|---|---|---|---|---|
| **OpenAI Agents SDK** | `Runner` loop over the dataset, or a handoff chain | `Agent` with `output_type=ResolverOutput` | sibling `Agent` with `output_type=ValidatorVerdict` | `Agent` that renders the report |
| **Google ADK** | `SequentialAgent` wrapping the question loop | `LlmAgent` (Resolver) | `LlmAgent` (Validator); both under a `LoopAgent` for the inner loop | `LlmAgent` for the summary |
| **LangGraph** | `StateGraph` node "route-next" with conditional edges | "resolve" node | "validate" node; conditional edge back to "resolve" on REVISE | "summarize" node |
| **CrewAI** | a `Crew` over all questions | `Agent` role "Examiner" | `Agent` role "Verifier" with a review `Task` | `Agent` role "Reporter" |

The inner loop is the natural showcase for each framework's distinctive
strength: LangGraph's conditional edges, ADK's `LoopAgent`, CrewAI's
task-delegation, and the OpenAI SDK's handoffs.

## Termination & failure handling

- **Inner loop** terminates on `accepted=True` or `attempts == MAX_ATTEMPTS`.
  On timeout, the last answer is kept and `raw.accepted=false`.
- **Outer loop** terminates when every question has a status entry. A question
  that errored (exception in the inner loop) is recorded with an empty response
  and `raw.error`, matching the existing `run.py` error contract.
- **Summary** always runs, even if some questions failed — the report's per-question
  table shows `accepted`/`attempts` so partial failures are visible.

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `MAX_ATTEMPTS` | 3 | Inner-loop cap before keeping the last answer. |
| `VALIDATOR_SEES_REFERENCE` | `false` | If `true` (ablation only), the Validator sees the reference answer — for measuring how much independence helps. |
| `ENABLE_VALIDATOR` | `true` | If `false`, runs the Resolver only — useful for comparing against the Phase 2/3 single-agent baseline. |

## Status & roadmap

- [x] Roles, loops, handoff models designed (this doc).
- [x] Shared contract + grader able to score `raw.attempts`/`raw.accepted`.
- [ ] **Phase 4 (LangGraph)**: first implementation of the full two-loop design,
      using a `StateGraph` with conditional edges for the inner loop.
- [ ] **Phase 5 (CrewAI)**: same roles via CrewAI tasks.
- [ ] Upgrade OpenAI Agents SDK and Google ADK implementations from their
      current single-agent baseline to the full two-loop design.
- [ ] Ablation study: single-agent baseline vs. resolver+validator vs.
      resolver+validator-with-reference.
