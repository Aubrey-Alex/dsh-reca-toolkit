"""Repair strategy router for the segment validator.

When the validator returns FAIL, this router (qwen-3.6-plus text-only LLM)
decides between THREE strategies:

  seed_reroll  — cheapest. Same segment.prompt, different seed. No LLM
                 call. Justified when round7 analysis showed that
                 attempt 2's score lift was mostly happyhorse seed
                 lottery (attempt 1 = 0.61 → attempt 2 = 0.66 → attempt
                 3 = 0.61 with identical motif failures across attempts).
                 Use when the prompt is judged sound and the failure
                 looks like a render-side draw (motif drifted but
                 prompt explicitly demands it; minor framing miss).

  micro_adjust — light. gpt-5.5 with SP memory rewrites JUST THIS
                 SEGMENT's prompt (+ optional end_state) with minimal
                 change. Preserves SP's original cinematic style. No
                 re-decomposition. Costs ~$0.01 + Wan call.

  replan       — heavy. gpt-5.5 with SP memory rewrites the WHOLE
                 shot's segments (may re-decompose, change segment
                 count). Expensive (~$0.03 + Wan calls per affected
                 seg) but recovers structural failures (identity drift
                 across multiple segs, story_goal mismatch,
                 decomposition itself wrong).

The router sees: the current segment.prompt, the latest judgment, and
the history of prior failure judgments for this segment. It outputs a
``RouterDecision`` with a 1-sentence rationale.

Design rules (iron):
  - Router is text-only (no image input), uses qwen-3.6-plus over DashScope.
  - Failures of the router (parse error, network) raise
    ``RouterError`` — pipeline defaults to ``micro_adjust`` as the
    conservative fallback.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from .validate import SegmentJudgment


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


Strategy = Literal["seed_reroll", "micro_adjust", "replan"]


from pathlib import Path as _Path
from videorlm.framework.templates import ChatPromptTemplate as _CPT

_PROMPTS_ROOT = _Path(__file__).resolve().parents[2] / "prompts"
ROUTER_SYSTEM_PROMPT: str = _CPT.from_files(
    system=_PROMPTS_ROOT / "validator_segment" / "router.md",
).format_system()


class RouterError(RuntimeError):
    """Raised when the router output is malformed or the call fails."""


@dataclass(frozen=True)
class RouterDecision:
    strategy: Strategy
    rationale: str


def _judgment_brief(j: SegmentJudgment) -> str:
    return (
        f"axis 分: aes={j.aesthetic_score:.2f} glob={j.global_alignment_score:.2f} "
        f"cons={j.action_consistency_score:.2f}; overall={j.overall_score:.2f}; "
        f"repair_action={j.repair_action}; reason={j.reason[:200]}; "
        f"edit_prompt={(j.edit_prompt or '<empty>')[:200]}"
    )


def route_repair_strategy(
    judgment: SegmentJudgment,
    current_seg: dict[str, Any],
    attempt_history: list[SegmentJudgment],
    router_cfg: Any,
) -> RouterDecision:
    """Ask the router LLM which repair strategy to use.

    Args:
      judgment: the latest validator judgment for the segment (FAIL).
      current_seg: current segment_spec dict (we read .segment_request.prompt).
      attempt_history: prior judgments for this segment (excludes ``judgment``).
      router_cfg: ``QwenConfig`` for the router (typically qwen3.6-plus,
        role=planner).

    Returns:
      A ``RouterDecision`` (strategy + rationale).

    Raises:
      ``RouterError`` if the LLM output can't be parsed.
    """
    from videorlm.backends.llm.agents import QwenAgent
    from videorlm.backends.llm.agents.base import AgentError

    sr = current_seg.get("segment_request") or {}
    current_prompt = sr.get("prompt", "")
    seg_id = current_seg.get("id") or sr.get("request_id") or "?"

    history_lines = "\n".join(
        f"  attempt {i+1}: {_judgment_brief(h)}"
        for i, h in enumerate(attempt_history)
    ) or "  (no prior attempts)"

    user = (
        f"segment_id: {seg_id}\n\n"
        f"[最新 validator 判定]\n{_judgment_brief(judgment)}\n\n"
        f"[当前 segment.prompt (你看看是否容易修)]\n{current_prompt[:1200]}\n\n"
        f"[之前失败历史]\n{history_lines}\n\n"
        f"请输出 strategy + rationale (JSON in ```json fence```)。"
    )

    cfg = router_cfg.with_overrides(system_prompt=ROUTER_SYSTEM_PROMPT)
    try:
        with QwenAgent(cfg) as agent:
            reply = agent.prompt(user)
    except AgentError as exc:
        raise RouterError(
            f"router {router_cfg.model} call failed for seg {seg_id}: "
            f"{type(exc).__name__}: {str(exc)[:300]}"
        ) from exc

    m = _FENCE_RE.search(reply)
    payload = m.group(1).strip() if m else reply.strip()
    try:
        d = json.loads(payload)
    except json.JSONDecodeError as e:
        raise RouterError(
            f"router output not parseable JSON: {e}; raw head: {reply[:200]!r}"
        ) from e

    strategy = str(d.get("strategy", "")).strip().lower()
    if strategy not in ("seed_reroll", "micro_adjust", "replan"):
        raise RouterError(
            f"router strategy must be one of 'seed_reroll' / 'micro_adjust' / "
            f"'replan', got {strategy!r}"
        )
    rationale = str(d.get("rationale", ""))[:400]
    return RouterDecision(strategy=strategy, rationale=rationale)  # type: ignore[arg-type]


__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "RouterDecision",
    "RouterError",
    "Strategy",
    "route_repair_strategy",
]
