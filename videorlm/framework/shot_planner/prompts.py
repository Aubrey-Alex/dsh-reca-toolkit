"""shot_planner system prompt.

Source of truth: ``framework/prompts/shot_planner/system.md``
This module loads that file at import time and re-exports the body as
``PARENT_SYSTEM_PROMPT`` for backwards-compat with existing callers
(``framework/shot_planner/plan.py``, ``framework/segment_planner/plan.py``,
``framework/_scripts/_freeze_prompts.py``).

There is no Python-level duplication of the prompt string anymore — edit
``prompts/shot_planner/system.md`` and import-time picks it up.
"""
from __future__ import annotations

from pathlib import Path

from videorlm.framework.templates import ChatPromptTemplate

_PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"

_TEMPLATE = ChatPromptTemplate.from_files(
    system=_PROMPTS_ROOT / "shot_planner" / "system.md",
)

PARENT_SYSTEM_PROMPT: str = _TEMPLATE.format_system()


__all__ = ["PARENT_SYSTEM_PROMPT"]
