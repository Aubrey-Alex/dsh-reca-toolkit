"""videorlm.framework — story prompt → planner output → render plan → mp4.

Usage
-----
```python
from videorlm.framework import run_planner, to_render_plan, run_render

story = open("input_examples/example_01.txt", encoding="utf-8").read()

# 1. Planning (parent agent + N forked children).
planner = run_planner(story, sp_cfg)

# 2. Convert to render plan (placeholder URLs).
render_plan = to_render_plan(planner, run_dir="run/example_01", seed=0)

# 3. Render side: single image DAG -> segments -> inline bridges -> concat.
final_mp4 = run_render(render_plan, out_path="run/example_01/final.mp4")
```

Each step is also exposed as a standalone function — see `pipeline` for
the full menu.
"""
from __future__ import annotations

from .pipeline import (
    ValidatorParams,
    SegmentValidatorParams,
    concat_final,
    merge_into_planner_output,
    parse_json_block,
    render_segments,
    run_planner,
    run_render,
    to_render_plan,
)
from .segment_planner import (
    SEGMENT_PLANNER_SYSTEM_PROMPT,
    format_segment_planner_user_prompt,
    make_segment_planner_from_parent_state,
    plan_segments_all,
    plan_segments_for_shot,
    reconstruct_parent_state,
    reconstruct_segment_planner_state,
)
from .shot_planner import PARENT_SYSTEM_PROMPT, plan_skeleton


__all__ = [
    # shot planner
    "PARENT_SYSTEM_PROMPT",
    "plan_skeleton",
    # segment planner (initial planning)
    "SEGMENT_PLANNER_SYSTEM_PROMPT",
    "format_segment_planner_user_prompt",
    "make_segment_planner_from_parent_state",
    "plan_segments_all",
    "plan_segments_for_shot",
    "reconstruct_parent_state",
    "reconstruct_segment_planner_state",
    # pipeline
    "ValidatorParams",
    "SegmentValidatorParams",
    "concat_final",
    "merge_into_planner_output",
    "parse_json_block",
    "render_segments",
    "run_planner",
    "run_render",
    "to_render_plan",
]
