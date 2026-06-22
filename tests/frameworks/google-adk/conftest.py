"""Make `frameworks/google-adk/` importable for these tests.

The framework folder uses a hyphen so it isn't a valid Python package name;
its modules use flat names (agent, output_models, solver, run). We insert the
folder at the FRONT of sys.path and clear any previously-imported flat modules
so that `import agent` resolves to THIS framework's modules, not another's.
"""
import sys
from pathlib import Path

_THIS_FW = "google-adk"
_FW_DIR = (
    Path(__file__).resolve().parents[3] / "frameworks" / _THIS_FW
)

# Remove cached flat modules from other frameworks so this dir's win.
_FLAT_MODULES = ("output_models", "agent", "solver", "run")
for _mod in list(sys.modules):
    if _mod in _FLAT_MODULES:
        del sys.modules[_mod]

sys.path.insert(0, str(_FW_DIR))
