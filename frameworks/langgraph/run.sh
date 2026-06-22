#!/usr/bin/env bash
# frameworks/langgraph/run.sh — self-contained entrypoint.
# Solves the shared dataset and writes result/results.json in the contract schema.
#
# Requires: OPENAI_API_KEY in the environment (or a .env file at repo root).
set -euo pipefail
cd "$(dirname "$0")"

REPO_ROOT="$(cd ../.. && pwd)"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

python run.py \
  --dataset "$REPO_ROOT/dataset/questions.json" \
  --output result/results.json
