"""validator/segment — VLM-driven judgment of rendered segment tail frames.

Mirrors validator/anchor but operates on R2V / I2V segment output instead
of i2i anchor PNG:
  - prompts.py : SEGMENT_VALIDATOR_SYSTEM_PROMPT (3 layers: end_state /
    motif / identity-drift; 6-field strict JSON schema)
  - validate.py : run_segment_validator() + apply_segment_repair()

Failure → at most ONE repair (RegenerateUnit / RepackPrompt / ReanchorState).
Never falls back to another backend — segment_validator is a retry layer,
not a fallback chain.
"""
from __future__ import annotations

from .prompts import SEGMENT_VALIDATOR_SYSTEM_PROMPT, format_segment_validator_user_prompt
from .validate import (
    SegmentJudgment,
    SegmentValidatorError,
    apply_segment_repair,
    run_segment_validator,
)


__all__ = [
    "SEGMENT_VALIDATOR_SYSTEM_PROMPT",
    "SegmentJudgment",
    "SegmentValidatorError",
    "apply_segment_repair",
    "format_segment_validator_user_prompt",
    "run_segment_validator",
]
