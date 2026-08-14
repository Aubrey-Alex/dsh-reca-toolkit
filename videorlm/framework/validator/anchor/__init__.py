"""validator/anchor — anchor visual judgment + image-edit repair loop.

Output schema (strict 7 fields) matches
videorlm/good_examples/validator_example.json:
  anchor_id / score / reason / pass_or_not / reference_inputs /
  negative_prompt / edit_prompt
"""
from __future__ import annotations

from ._codex_vision import make_codex_vision_call
from .prompts import VALIDATOR_SYSTEM_PROMPT, format_validator_user_prompt
from .validate import (
    REQUIRED_FIELDS,
    ValidatorError,
    apply_image_edit_repair,
    make_validator_from_segment_planner_state,
    validate_anchor,
    validate_anchor_via_agent,
    validate_and_repair,
)


__all__ = [
    "VALIDATOR_SYSTEM_PROMPT",
    "format_validator_user_prompt",
    "validate_anchor",
    "apply_image_edit_repair",
    "validate_and_repair",
    "make_validator_from_segment_planner_state",
    "validate_anchor_via_agent",
    "make_codex_vision_call",
    "ValidatorError",
    "REQUIRED_FIELDS",
]