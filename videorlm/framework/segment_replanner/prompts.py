"""segment_replanner system prompts + per-task user prompt builders.

Two personas, both fork from segment_planner state and inherit its
conversation memory:

  SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT — full-shot rewrite agent.
    Inherits [user(story), assistant(skeleton)] (2 msgs, first_pair_only).
    Rewrites ALL segments of a shot given user_feedback. Can change
    segment count (split / merge) and re-decompose. Used for structural
    failure (identity lost across multiple segments, story_goal drift).

  SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT — single-segment touch-up agent.
    Inherits [story, skeleton, sp_user, segments_for_shot] (4 msgs)
    so the agent sees its own prior SP output for THIS shot. Rewrites
    ONLY the failing segment's prompt + end_state with minimal-diff
    style preservation. Used for local fails (one motif missed,
    one composition tweak).

Both system prompts are loaded from
``framework/prompts/segment_replanner/system_*.md``. The two runtime
user-prompt builders below stay in Python — they're dynamic assembly
(JSON dump of original_segments, judgment scores, planner asset enum),
not static templates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from videorlm.framework.templates import ChatPromptTemplate

_PROMPTS_ROOT = Path(__file__).resolve().parents[1] / "prompts"

_REPLAN_TEMPLATE = ChatPromptTemplate.from_files(
    system=_PROMPTS_ROOT / "segment_replanner" / "system_replan.md",
)
_MICRO_TEMPLATE = ChatPromptTemplate.from_files(
    system=_PROMPTS_ROOT / "segment_replanner" / "system_micro_adjust.md",
)

SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT: str = _REPLAN_TEMPLATE.format_system()
SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT: str = _MICRO_TEMPLATE.format_system()


def format_segment_replan_user_prompt(
    shot: dict[str, Any],
    anchor: dict[str, Any],
    planner: dict[str, Any] | None,
    original_segments: dict[str, Any],
    user_feedback: str,
) -> str:
    """Build the user turn for SP replan mode (driven by validator/human feedback).

    Used together with SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT, NOT with the
    original SEGMENT_PLANNER_SYSTEM_PROMPT.
    """
    portraits = ""
    place_list = ""
    props = ""
    if planner:
        portraits = ", ".join(p["id"] for p in planner.get("portrait_plan", []))
        prop_list: list[str] = []
        for loc in planner.get("location_plan", {}).values():
            prop_list.extend((loc.get("props") or {}).keys())
        place_list = ", ".join(planner.get("location_plan", {}).keys())
        props = ", ".join(prop_list)

    original_json = json.dumps(original_segments, ensure_ascii=False, indent=2)

    return f"""shot id = {shot["id"]}, duration_s = {shot["duration_s"]}s

[shot 设计 — 只读, 不要改]
- start_state (T=0):    {shot["start_state"]}
- end_state   (T={shot["duration_s"]}s): {shot["end_state"]}
- visual_intent:        {shot["visual_intent"]}
- anchor.prompt (已渲染, 不要改):
{anchor["prompt"]}

[原 segments — 上一轮 SP 的输出, 重写它]
```json
{original_json}
```

[人工反馈 — 必须接受为权威意见]
{user_feedback}

[可选 reference_inputs ID]
- portrait: {portraits}
- place: {place_list}
- prop: {props}

任务: 综合上述 (shot 设计 + 原 segments + 反馈 + 资产 ID), 重写本 shot 的 segments 字典。
保持 ∑ duration_s = {shot["duration_s"]}, 单段 ∈ [3, 15]。段数 (segments_count_in_shot) 可变。
仍按 system_prompt 的 5 步流程 + R2V 物理性写, 跑 system_prompt 的 6 步 self-check。
仅输出 JSON segments dict, 包在 ```json ... ``` fence 中。
"""


def format_micro_adjust_user_prompt(
    segment_id: str,
    current_seg: dict[str, Any],
    judgment: Any,
) -> str:
    """Build the user turn for the micro-adjust engineer.

    Args:
      segment_id: failing segment's id.
      current_seg: segment_spec dict (must have ``segment_request.prompt``
        and ``end_state``).
      judgment: ``SegmentJudgment`` from the validator.

    Returns:
      A user-message text that summarizes the validator failure +
      shows the current prompt/end_state + asks for a JSON revision.
    """
    sr = current_seg.get("segment_request") or {}
    current_prompt = sr.get("prompt", "")
    current_end_state = current_seg.get("end_state", "")
    return (
        f"本次微调任务: validator 发现 segment `{segment_id}` 渲染失败, 请你按 system_prompt 中的"
        f"改动边界做最小幅度微调。\n\n"
        f"[validator 评分]\n"
        f"- aesthetic_score:        {judgment.aesthetic_score:.2f} (权重 0.25)\n"
        f"- global_alignment_score: {judgment.global_alignment_score:.2f} (权重 0.40)\n"
        f"- action_consistency_score: {judgment.action_consistency_score:.2f} (权重 0.35)\n"
        f"- overall_score:          {judgment.overall_score:.2f} (pass 阈 ≥0.7, 每 axis ≥0.6)\n\n"
        f"[validator 给的失败原因]\n{judgment.reason}\n\n"
        f"[validator 给的修复建议 (可参考, 不要照抄)]\n{judgment.edit_prompt or '(空)'}\n\n"
        f"[当前 segment.prompt (你之前写的)]\n{current_prompt}\n\n"
        f"[当前 segment.end_state (你之前写的)]\n{current_end_state}\n\n"
        f"输出 严格 JSON, 仅含 segment `{segment_id}` 的微调结果 "
        f"({{<sid>: {{prompt, end_state, _change_summary}}}}), 包在 ```json ... ``` fence。"
    )


__all__ = [
    "SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT",
    "SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT",
    "format_segment_replan_user_prompt",
    "format_micro_adjust_user_prompt",
]
