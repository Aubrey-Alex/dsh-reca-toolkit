"""Role-based concurrency pools for the framework.

Four named pools route LLM/render concurrency by **role** instead of by
underlying model. This keeps every planner — regardless of model
(qwen3.6-max-preview, gpt-5.5, etc.) — sharing one budget, validators
split into anchor (text) vs segment (vision), and a separate render pool
guarding the shot-chain executor.

  - PLANNER_POOL_SIZE          (RECA_PLANNER_POOL_SIZE,          default 8)
      Shared by shot_planner (parent) + segment_planner +
      segment_replan. The three are temporally non-overlapping in normal
      flow (shot_planner finishes before segment_planner fan-out runs;
      segment_replan only fires on human feedback), so a single pool of
      8 services all three without starvation. The fan-out (~18 shots in
      parallel for a long video) is the only step that actually fills the
      pool.

  - ANCHOR_VALIDATOR_POOL_SIZE (RECA_ANCHOR_VALIDATOR_POOL_SIZE, default 4)
      Anchor validator (gpt-5.5 text on OpenAI-compat gateway). Text
      calls are fast; 4 concurrency is healthy on the gateway TPS.

  - SEGMENT_VALIDATOR_POOL_SIZE(RECA_SEGMENT_VALIDATOR_POOL_SIZE, default 2)
      Segment validator (qwen3-vl-plus VL on DashScope). VL calls are
      slow and expensive; each segment may need up to 3 calls
      (validate + repair × 2). 2 concurrency avoids saturating the
      qwen3-vl-plus TPS budget.

  - RENDER_POOL_SIZE           (RECA_RENDER_POOL_SIZE,           default 8)
      Caps the inner shot-chain executor in `pipeline.render_segments`.
      The per-backend KeyPool inside each VideoSegmentBackend is the
      *real* GPU-slot throttle; this pool just bounds how many shot
      chains are "in-flight" through the framework at once.

All four are read from env at import time. Override per-run via env
without touching code:

    RECA_PLANNER_POOL_SIZE=12 RECA_ANCHOR_VALIDATOR_POOL_SIZE=4 \
    RECA_SEGMENT_VALIDATOR_POOL_SIZE=2 RECA_RENDER_POOL_SIZE=6 \
    python -m videorlm.framework._scripts._smoke ...
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(1, v)


PLANNER_POOL_SIZE: int = _env_int("RECA_PLANNER_POOL_SIZE", 8)
ANCHOR_VALIDATOR_POOL_SIZE: int = _env_int("RECA_ANCHOR_VALIDATOR_POOL_SIZE", 4)
SEGMENT_VALIDATOR_POOL_SIZE: int = _env_int("RECA_SEGMENT_VALIDATOR_POOL_SIZE", 2)
RENDER_POOL_SIZE: int = _env_int("RECA_RENDER_POOL_SIZE", 8)


PLANNER_ROLE = "planner"
ANCHOR_VALIDATOR_ROLE = "anchor_validator"
SEGMENT_VALIDATOR_ROLE = "segment_validator"
RENDER_ROLE = "render"


def pool_size_for_role(role: str) -> int:
    """Map role string to its env-driven pool size. Returns 8 if unknown."""
    if role == PLANNER_ROLE:
        return PLANNER_POOL_SIZE
    if role == ANCHOR_VALIDATOR_ROLE:
        return ANCHOR_VALIDATOR_POOL_SIZE
    if role == SEGMENT_VALIDATOR_ROLE:
        return SEGMENT_VALIDATOR_POOL_SIZE
    if role == RENDER_ROLE:
        return RENDER_POOL_SIZE
    return 8


__all__ = [
    "PLANNER_POOL_SIZE",
    "ANCHOR_VALIDATOR_POOL_SIZE",
    "SEGMENT_VALIDATOR_POOL_SIZE",
    "RENDER_POOL_SIZE",
    "PLANNER_ROLE",
    "ANCHOR_VALIDATOR_ROLE",
    "SEGMENT_VALIDATOR_ROLE",
    "RENDER_ROLE",
    "pool_size_for_role",
]
