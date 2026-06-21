"""Test config: make framework folders with hyphens importable.

Framework folders like `openai-agents` and `google-adk` use hyphens and so
aren't valid Python package names. Their modules use flat names
(output_models, agent, solver, run) and are imported by inserting the
framework dir onto sys.path here.
"""
import sys
from pathlib import Path

_FRAMEWORKS = Path(__file__).resolve().parent.parent / "frameworks"
for _sub in ("openai-agents", "google-adk"):
    _dir = _FRAMEWORKS / _sub
    if _dir.is_dir():
        sys.path.insert(0, str(_dir))
