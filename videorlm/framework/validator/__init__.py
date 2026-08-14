"""validator subpackage — anchor + segment visual judgment.

Two distinct judgment lanes:
  - `validator.anchor` — anchor PNG (i2i output) judgment + image-edit repair
  - `validator.segment` — rendered video segment tail-frame judgment +
    RegenerateUnit / RepackPrompt repair (single retry, no fallback chain)

The top-level re-exports below keep legacy callers using `from
videorlm.framework.validator import VALIDATOR_SYSTEM_PROMPT` working —
they implicitly resolve to the anchor variant (the old default).
"""
from __future__ import annotations

from .anchor import (
    REQUIRED_FIELDS,
    VALIDATOR_SYSTEM_PROMPT,
    ValidatorError,
    apply_image_edit_repair,
    format_validator_user_prompt,
    make_codex_vision_call,
    make_validator_from_segment_planner_state,
    validate_and_repair,
    validate_anchor,
    validate_anchor_via_agent,
)
from .segment import (
    SEGMENT_VALIDATOR_SYSTEM_PROMPT,
    SegmentJudgment,
    SegmentValidatorError,
    apply_segment_repair,
    run_segment_validator,
)


__all__ = [
    # anchor — original public API
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
    # segment — new
    "SEGMENT_VALIDATOR_SYSTEM_PROMPT",
    "SegmentJudgment",
    "SegmentValidatorError",
    "apply_segment_repair",
    "run_segment_validator",
]