"""segment_replanner core logic — feedback-driven segment revision.

Two strategies, both fork from segment_planner state:

  replan_segments_for_shot(...) — heavy. Full-shot rewrite given
    user_feedback. Used when validator flags structural failure that
    needs re-decomposition.

  micro_adjust_single_segment(...) — light. Single-segment touch-up
    given validator judgment. Used when failure is local and the SP's
    existing decomposition is still sound.

After framework-v2 (2026-05-16) both retry loops route through
``framework/parsers.py::retry_agent_call`` (single source of truth for
"prompt → parse → on-fail tail-cleanup → retry"). The replan branch
validates the new segments against the parent shot via
``ShotSegments.validate_for_shot`` (Pydantic cross-field rules). The
micro-adjust branch ships its own tiny parse fn since the response
schema (``{<sid>: {prompt, end_state, _change_summary}}``) is too
specific to deserve its own Pydantic model.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from videorlm.framework.parsers import ParseError, retry_agent_call, strip_markdown_fence
from videorlm.framework.schemas import BoundaryAnchor, Shot, ShotSegments

from .prompts import (
    SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT,
    SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT,
    format_micro_adjust_user_prompt,
    format_segment_replan_user_prompt,
)

# ``videorlm.backends.llm.agents`` is imported lazily inside functions
# because it transitively requires the vendored ``project`` module on
# sys.path, which entrypoints (``_smoke.py::setup_path``) inject only
# after argparse runs.


# ── REPLAN (whole-shot rewrite, feedback-driven) ────────────────────────


def make_segment_planner_replan_from_parent_state(
    parent_state: dict[str, Any],
    sp_replan_cfg: Any,
) -> Any:
    """Like make_segment_planner_from_parent_state but uses REPLAN system prompt.

    Forks from parent.state and inherits the parent's [user(story),
    assistant(skeleton)] tail (first_pair_only).
    """
    from videorlm.backends.llm.agents import QwenAgent
    from videorlm.framework._common.fork import sub_conversation_with_system_swap

    new_state = sub_conversation_with_system_swap(
        parent_state, SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT,
        inherit_policy="first_pair_only",
    )
    return QwenAgent(sp_replan_cfg, state=new_state)


def replan_segments_for_shot(
    parent_state: dict[str, Any],
    shot: dict[str, Any],
    anchor: dict[str, Any],
    skeleton: dict[str, Any] | None,
    sp_replan_cfg: Any,
    *,
    original_segments: dict[str, Any],
    user_feedback: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Re-plan one shot's segments based on validator/human feedback.

    Independent of plan_segments_for_shot:
      - uses SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT (separate persona)
      - takes original_segments + user_feedback, asks SP to rewrite
      - shot's start/end/visual_intent/duration/anchor.prompt are NOT
        changed (those belong to shot_planner; this is segment-level only)

    Retries up to ``max_retries`` via ``retry_agent_call`` on JSON parse /
    cross-field validation failures.

    Returns the new segments dict (same schema as plan_segments_for_shot).
    """
    print(
        f"[agent-trace] sp-replan for shot={shot['id']} build: feedback={len(user_feedback)} chars, "
        f"original_segs={len(original_segments)} (system swapped to SEGMENT_PLANNER_REPLAN_SYSTEM_PROMPT)",
        flush=True,
    )

    shot_pyd = Shot.model_validate(shot)
    anchor_pyd = BoundaryAnchor.model_validate(anchor)

    def _parse(reply: str) -> dict[str, Any]:
        body = strip_markdown_fence(reply)
        raw = json.loads(body)
        ShotSegments.validate_for_shot(raw, shot_pyd, anchor_pyd)
        return raw

    def _on_fail(label: str, attempt: int, exc: BaseException, reply_head: str) -> None:
        print(
            f"[sp-replan shot={shot['id']}] attempt {attempt}/{max_retries} parse/validate failed "
            f"({type(exc).__name__}: {str(exc)[:160]}); reply head: {reply_head!r}",
            flush=True,
        )

    sp = make_segment_planner_replan_from_parent_state(parent_state, sp_replan_cfg)
    user_prompt = format_segment_replan_user_prompt(
        shot, anchor, skeleton, original_segments, user_feedback,
    )
    try:
        with sp:
            segs = retry_agent_call(
                sp, user_prompt, _parse,
                max_retries=max_retries,
                sleep_base_s=2.0, sleep_cap_s=60.0,
                on_fail=_on_fail,
                label=f"sp-replan shot={shot['id']}",
            )
            after = len(sp.state.get("messages", []) or [])
            print(
                f"[agent-trace] sp-replan for shot={shot['id']} done: messages={after}",
                flush=True,
            )
            return segs
    except ParseError as exc:
        last = exc.last_exc or exc
        raise RuntimeError(
            f"replan_segments_for_shot[{shot['id']}]: exhausted {max_retries} retries; "
            f"last={type(last).__name__}: {str(last)[:200]}"
        ) from exc


# ── MICRO_ADJUST (single-segment touch-up) ──────────────────────────────


def make_micro_adjust_from_sp_state(
    sp_state: dict[str, Any],
    ma_cfg: Any,
) -> Any:
    """Fork a micro-adjust engineer agent from the SP post-turn state.

    Inherits 4-msg tail [user(story), assistant(skeleton), user(per-shot),
    assistant(segments_for_shot)] — i.e. the FULL conversation the SP had
    when it wrote segments for this shot. The agent then sees its own
    prior output and can do a minimal-diff revision on a single segment.

    System prompt is swapped to ``SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT``,
    NOT the SP system or the SP_REPLAN system — the persona is a
    "微调工程师" focused on small, in-style edits.
    """
    from videorlm.backends.llm.agents import QwenAgent
    from videorlm.framework._common.fork import sub_conversation_with_system_swap

    new_state = sub_conversation_with_system_swap(
        sp_state,
        SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT,
        inherited_msg_count=4,
    )
    return QwenAgent(ma_cfg, state=new_state)


def micro_adjust_single_segment(
    sp_state: dict[str, Any],
    current_seg: dict[str, Any],
    judgment: Any,
    ma_cfg: Any,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Single-segment micro-adjust via gpt-5.5 with SP memory.

    Args:
      sp_state: the segment_planner's POST-turn state for THIS shot
        (must contain story + skeleton + sp_user + segments tail).
      current_seg: the failing segment_spec dict (must have
        ``segment_request.prompt`` and ``end_state``).
      judgment: ``SegmentJudgment`` from the validator.
      ma_cfg: ``QwenConfig`` for the micro-adjust agent (typically
        gpt-5.5, same provider as segment_planner).

    Returns:
      A new segment_spec dict with revised ``segment_request.prompt`` and
      (optionally) revised ``end_state``. All other fields (id, refs,
      duration, anchor) are preserved from ``current_seg``.

    Raises:
      ``RuntimeError`` after ``max_retries`` JSON parse / schema failures.
    """
    sid = current_seg.get("id") or (current_seg.get("segment_request") or {}).get("request_id") or "?"
    user_prompt = format_micro_adjust_user_prompt(sid, current_seg, judgment)

    print(
        f"[agent-trace] micro_adjust for {sid} build: inherited_tail=4 "
        "(system swapped to SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT)",
        flush=True,
    )

    def _parse(reply: str) -> dict[str, Any]:
        """Parse + minimal schema check for the micro-adjust reply:
        ``{<sid>: {prompt, end_state, _change_summary}}``."""
        body = strip_markdown_fence(reply)
        revised = json.loads(body)
        if sid not in revised:
            raise KeyError(
                f"micro-adjust output missing key {sid}; got {list(revised)[:3]}"
            )
        entry = revised[sid]
        new_prompt = entry.get("prompt", "")
        if not new_prompt or not new_prompt.strip():
            raise ValueError(f"micro-adjust returned empty prompt for {sid}")
        new_end_state = entry.get("end_state") or current_seg.get("end_state", "")
        change_summary = str(entry.get("_change_summary", ""))[:200]

        new_spec = copy.deepcopy(current_seg)
        new_sr = new_spec.setdefault("segment_request", {})
        new_sr["prompt"] = new_prompt
        new_spec["end_state"] = new_end_state
        new_spec["_micro_adjust_change_summary"] = change_summary
        return new_spec

    def _on_fail(label: str, attempt: int, exc: BaseException, reply_head: str) -> None:
        print(
            f"[micro-adjust {sid}] attempt {attempt}/{max_retries} parse/validate failed "
            f"({type(exc).__name__}: {str(exc)[:160]}); reply head: {reply_head!r}",
            flush=True,
        )

    agent = make_micro_adjust_from_sp_state(sp_state, ma_cfg)
    try:
        with agent:
            new_spec = retry_agent_call(
                agent, user_prompt, _parse,
                max_retries=max_retries,
                sleep_base_s=2.0, sleep_cap_s=60.0,
                on_fail=_on_fail,
                label=f"micro-adjust {sid}",
            )
            new_prompt = (new_spec.get("segment_request") or {}).get("prompt", "")
            change_summary = new_spec.get("_micro_adjust_change_summary", "")
            print(
                f"[agent-trace] micro_adjust for {sid} done: "
                f"prompt_len {len(current_seg.get('segment_request',{}).get('prompt','') or '')} "
                f"→ {len(new_prompt)}, change: {change_summary[:100]}",
                flush=True,
            )
            return new_spec
    except ParseError as exc:
        last = exc.last_exc or exc
        raise RuntimeError(
            f"micro_adjust_single_segment[{sid}]: exhausted {max_retries} retries; "
            f"last={type(last).__name__}: {str(last)[:200]}"
        ) from exc


__all__ = [
    "make_segment_planner_replan_from_parent_state",
    "make_micro_adjust_from_sp_state",
    "replan_segments_for_shot",
    "micro_adjust_single_segment",
]
