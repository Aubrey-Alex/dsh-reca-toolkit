"""segment_planner subpackage — Option B fork from parent state.

Mirrors the layout of `framework/validator`:
  - `prompts.py`    : SEGMENT_PLANNER_SYSTEM_PROMPT + per-shot user prompt formatter
  - `plan.py`       : make_segment_planner_from_parent_state + plan_* functions

Public API (initial-planning only):
  SEGMENT_PLANNER_SYSTEM_PROMPT      — system prompt for the SP agent
  format_segment_planner_user_prompt — build the per-shot user message body
  make_segment_planner_from_parent_state — fork(parent_state) with system swap
  plan_segments_for_shot             — single-shot turn (build → prompt → parse)
  plan_segments_all                  — fan-out across shots, ThreadPoolExecutor
  reconstruct_segment_planner_state  — offline rebuild for validator (after SP closed)
  reconstruct_parent_state           — offline rebuild parent state for replan path
"""
from __future__ import annotations

from .plan import (
    make_segment_planner_from_parent_state,
    plan_segments_all,
    plan_segments_for_shot,
    reconstruct_parent_state,
    reconstruct_segment_planner_state,
)
from .prompts import (
    SEGMENT_PLANNER_SYSTEM_PROMPT,
    format_segment_planner_user_prompt,
)


__all__ = [
    "SEGMENT_PLANNER_SYSTEM_PROMPT",
    "format_segment_planner_user_prompt",
    "make_segment_planner_from_parent_state",
    "plan_segments_for_shot",
    "plan_segments_all",
    "reconstruct_parent_state",
    "reconstruct_segment_planner_state",
]
