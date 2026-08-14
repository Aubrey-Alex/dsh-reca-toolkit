"""segment_replanner package.

Validator-feedback / human-feedback driven segment revision agents.
Forks from segment_planner state (inherits its conversation memory) but
swaps system prompt to one of two personas:

  - SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT — whole-shot rewrite
  - SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT  — single-segment touch-up

Both routes are driven by the router (``validator/segment/router.py``)
which decides between ``replan`` and ``micro_adjust`` per failure.
"""
from .plan import (
    make_micro_adjust_from_sp_state,
    make_segment_planner_replan_from_parent_state,
    micro_adjust_single_segment,
    replan_segments_for_shot,
)
from .prompts import (
    SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT,
    SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT,
    format_micro_adjust_user_prompt,
    format_segment_replan_user_prompt,
)

__all__ = [
    "SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT",
    "SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT",
    "format_segment_replan_user_prompt",
    "format_micro_adjust_user_prompt",
    "make_segment_planner_replan_from_parent_state",
    "make_micro_adjust_from_sp_state",
    "replan_segments_for_shot",
    "micro_adjust_single_segment",
]
