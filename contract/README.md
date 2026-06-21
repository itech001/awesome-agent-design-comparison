# Contract

The shared agreement between the dataset, the frameworks, and the grader.

## Schemas

- **`question.schema.json`** — one question in `dataset/questions.json`.
- **`result.schema.json`** — a framework's output file at
  `frameworks/<name>/result/results.json`.

## Validating

```bash
python contract/validate.py --file dataset/questions.json --schema question
python contract/validate.py --file frameworks/openai-agents/result/results.json --schema result
```

Exit code 0 means valid.

> Note: the schemas validate a single object. `dataset/questions.json` is an
> array; the dataset tests in `tests/test_dataset.py` validate each item. To
> validate the whole dataset, run `pytest tests/test_dataset.py`.

## Rules for frameworks

1. Read `dataset/questions.json` (read-only). Never copy or redefine the schema.
2. Write `result/results.json` in exactly `result.schema.json` shape.
3. Top-level field names and types are fixed. Use the `raw` object for any
   framework-specific detail (traces, tool calls, agent state).
4. `reasoning` is separate from `response` so the grader scores the final answer
   but can also analyze reasoning quality later.
