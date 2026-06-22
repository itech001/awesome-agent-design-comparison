# Dataset

The 50-question shared dataset. Every framework reads this file (read-only);
every framework writes results that the grader compares against it.

## File

`questions.json` — a JSON array of 50 question objects. Each conforms to
`contract/question.schema.json`.

## Distribution

| Subject | Multiple choice | Short answer | Total |
|---|---|---|---|
| chinese | 5 | 3 | 8 |
| math | 4 | 4 | 8 |
| english | 5 | 3 | 8 |
| physics | 4 | 2 | 6 |
| chemistry | 4 | 2 | 6 |
| biology | 4 | 2 | 6 |
| history | 3 | 1 | 4 |
| geography | 3 | 1 | 4 |
| **Total** | **32** | **18** | **50** |

## Fields

See `contract/question.schema.json` for the authoritative schema and
`contract/README.md` for the human-readable explanation.

Key fields:
- `id` — `<subject>-<seq>`, globally unique.
- `answer_type` — `letter` (MC, exact match) or `text` (rubric).
- `scoring` — drives grader behavior: `exact` for MC, `rubric` for short answer.
- `source` — original senior secondary school entrance exam paper the question is based on.
- `notes` — accepted-answer hints for the grader (synonyms, equivalent forms).

## Validation

```bash
# Validate the whole dataset (each question against the schema, plus structural checks):
pytest tests/test_dataset.py

# Validate a single object:
python contract/validate.py --file <some-question.json> --schema question
```
