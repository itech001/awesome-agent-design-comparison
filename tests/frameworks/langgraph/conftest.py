"""Make `frameworks/langgraph/` importable for these tests.

LangGraph's folder has no hyphen, but we keep flat imports for consistency
with the other frameworks' test-isolation pattern (each framework's tests live
in tests/frameworks/<name>/ with a conftest that puts that framework's dir on
sys.path and clears cached flat modules).
"""
import sys
from pathlib import Path

_FW_DIR = Path(__file__).resolve().parents[3] / "frameworks" / "langgraph"
_FLAT_MODULES = ("output_models", "llm", "state", "graph", "solver", "run")
for _mod in list(sys.modules):
    if _mod in _FLAT_MODULES:
        del sys.modules[_mod]
sys.path.insert(0, str(_FW_DIR))
