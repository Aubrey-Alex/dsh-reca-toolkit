"""segment_planner system prompt + per-shot user prompt builder.

System prompt source of truth: ``framework/prompts/segment_planner/system.md``
This module loads that file at import time and re-exports it as
``SEGMENT_PLANNER_SYSTEM_PROMPT`` for backwards-compat with existing
callers (``framework/segment_planner/plan.py``,
``framework/_scripts/_freeze_prompts.py``).

The runtime helper ``format_segment_planner_user_prompt`` stays in
Python — it's dynamic prompt assembly (per-shot interpolation of
``shot.id`` / ``shot.duration_s`` / ``anchor.prompt`` / planner asset
IDs), not a static template.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from videorlm.framework.templates import ChatPromptTemplate

_PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"

_TEMPLATE = ChatPromptTemplate.from_files(
    system=_PROMPTS_ROOT / "segment_planner" / "system.md",
)

SEGMENT_PLANNER_SYSTEM_PROMPT: str = _TEMPLATE.format_system()


def format_segment_planner_user_prompt(
    shot: dict[str, Any],
    anchor: dict[str, Any],
    planner: dict[str, Any] | None = None,
) -> str:
    """Build the per-shot user message body.

    High-priority constraints first (硬约束 → 时间链 → anchor 第一帧),
    contextual info last (story_goal → asset enum). `planner` optional —
    when provided, enumerates available portrait/place/prop IDs so the
    SP can pick valid reference_inputs without traversing fork history.

    Note: SEGMENT_PLANNER_SYSTEM_PROMPT is NOT prepended here — it lives
    in messages[0] (system) of the segment_planner agent.
    """
    body = f"""shot id = {shot["id"]}, duration_s = {shot["duration_s"]}s

[硬约束]
- ∑ segment.duration_s = {shot["duration_s"]}, 单段 ∈ [3, 15]
- segment 0 first_frame = anchor (id={anchor["id"]})
- segment k≥1 first_frame = previous segment 尾帧

[时间链端点]
- start_state (T=0):    {shot["start_state"]}
- end_state   (T={shot["duration_s"]}s): {shot["end_state"]}

[anchor 第一帧画面]
{anchor["prompt"]}

[镜头运动 + 风格 + 氛围 — 所有 segment 共享]
{shot["visual_intent"]}

[功能/意义 — 背景参考]
{shot["story_goal"]}
"""
    if planner:
        portraits = ", ".join(p["id"] for p in planner.get("portrait_plan", []))
        props: list[str] = []
        for loc in planner.get("location_plan", {}).values():
            props.extend((loc.get("props") or {}).keys())
        place_list = ", ".join(planner.get("location_plan", {}).keys())
        body += f"""
[可选 reference_inputs ID]
- portrait: {portraits}
- place: {place_list}
- prop: {", ".join(props)}
"""
    return body


__all__ = [
    "SEGMENT_PLANNER_SYSTEM_PROMPT",
    "format_segment_planner_user_prompt",
]
