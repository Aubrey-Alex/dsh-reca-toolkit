"""shot_planner subpackage — parent agent that produces the skeleton.

Mirrors the layout of `framework/segment_planner`:
  - `prompts.py`  : PARENT_SYSTEM_PROMPT
  - `plan.py`     : plan_skeleton + _validate_skeleton
"""
from __future__ import annotations

from .plan import plan_skeleton
from .prompts import PARENT_SYSTEM_PROMPT


__all__ = [
    "PARENT_SYSTEM_PROMPT",
    "plan_skeleton",
]
