"""shot_planner core: ``plan_skeleton`` — story → skeleton JSON.

After the framework-v2 refactor (2026-05-16), the JSON parse + schema
validate + agent retry loop all live in framework/parsers.py and
framework/schemas.py. This module is just the thin entry point that:
  1. constructs a ``JSONOutputParser(Skeleton)``
  2. calls ``parse_with_retry(parent, story)``
  3. dumps the validated Pydantic model back to ``dict`` for the
     existing callers (pipeline.run_render, _smoke.py, etc.)

``_validate_skeleton`` is kept as a thin Pydantic-backed wrapper so
existing imports (e.g. ``from .plan import _validate_skeleton`` for
ad-hoc smoke scripts) keep working.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from videorlm.framework.parsers import JSONOutputParser, ParseError
from videorlm.framework.schemas import Skeleton

from .prompts import PARENT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from videorlm.backends.llm.agents import Agent


_PARSER: JSONOutputParser[Skeleton] = JSONOutputParser(Skeleton, name="plan_skeleton")


def _log_agent_state(label: str, agent: Any) -> None:
    msgs = agent.state.get("messages", []) or []
    summary = " | ".join(f"{m.role}({len(m.content)} chars)" for m in msgs)
    print(f"[agent-trace] {label}: messages={len(msgs)} [{summary}]", flush=True)


def _on_parse_fail(label: str, attempt: int, exc: BaseException, reply_head: str) -> None:
    """Match the legacy stdout format so existing log-grep tooling
    (`grep "plan_skeleton" *.log`) keeps working."""
    print(
        f"[plan_skeleton] attempt {attempt} parse/validate failed "
        f"({type(exc).__name__}: {str(exc)[:160]}); reply head: {reply_head!r}",
        flush=True,
    )


def plan_skeleton(
    parent: "Agent",
    story: str,
    *,
    max_retries: int = int(os.environ.get("RECA_PLAN_SKELETON_MAX_RETRIES", "10")),
) -> dict[str, Any]:
    """Send raw story to a started parent agent and return the skeleton
    as a plain ``dict`` (same legacy interface).

    Parse + retry + tail-cleanup logic is centralised in
    ``framework/parsers.py::retry_agent_call``; schema rules
    (``len(shots) == len(boundary_anchors)``, ``transitions == shots-1``,
    duration ranges) live in ``framework/schemas.py::Skeleton``.

    Failures after ``max_retries`` raise ``RuntimeError`` (same as legacy)
    with the last underlying exception attached.
    """
    _log_agent_state(f"parent before plan_skeleton (max_retries={max_retries})", parent)
    try:
        skeleton: Skeleton = _PARSER.parse_with_retry(
            parent,
            story,
            max_retries=max_retries,
            sleep_base_s=2.0,
            sleep_cap_s=60.0,
            on_parse_fail=_on_parse_fail,
        )
    except ParseError as exc:
        last = exc.last_exc or exc
        raise RuntimeError(
            f"plan_skeleton: exhausted {max_retries} retries; "
            f"last={type(last).__name__}: {str(last)[:200]}"
        ) from exc
    _log_agent_state("parent after plan_skeleton", parent)
    # Preserve the legacy dict-return contract so pipeline.run_render and
    # other consumers don't need updating in the same PR.
    return skeleton.model_dump()


def _validate_skeleton(s: dict[str, Any]) -> None:
    """Thin Pydantic-backed shim for backwards-compat with ad-hoc imports.

    Equivalent to ``Skeleton.model_validate(s)`` but raises ``ValueError``
    (the original API) instead of Pydantic's ``ValidationError`` so
    callers that catch the legacy exception type keep working.
    """
    from pydantic import ValidationError
    try:
        Skeleton.model_validate(s)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


__all__ = ["plan_skeleton", "_validate_skeleton", "PARENT_SYSTEM_PROMPT"]
