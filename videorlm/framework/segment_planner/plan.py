"""segment_planner core logic — Option B fork from parent state.

Architecture (mirrors validator):
  parent state.messages = [system(PARENT), user(story), assistant(skeleton)]
                                |
                                | make_segment_planner_from_parent_state(parent.state, sp_cfg)
                                ▼
  sp state.messages     = [system(SP),     user(story), assistant(skeleton)]
                                |
                                | sp.prompt(per-shot user message)
                                ▼
  sp state.messages     = [system(SP), user(story), assistant(skeleton),
                           user(per-shot), assistant(segments_for_shot)]

Key API:
  - ``make_segment_planner_from_parent_state(parent_state, sp_cfg)`` — build
    a fresh SP agent for one shot, NOT yet started.
  - ``plan_segments_for_shot(parent_state, shot, anchor, skeleton, sp_cfg)`` —
    full per-shot turn (build SP → prompt → parse → validate). Closes the
    SP before return.
  - ``plan_segments_all(parent_state, skeleton, sp_cfg, max_workers=8)`` —
    fan-out all shots in a thread pool.
  - ``reconstruct_segment_planner_state(planner, story, shot_id)`` — rebuild
    the post-turn state offline (used by validator after SP closed).

After framework-v2 (2026-05-16), the JSON parse + schema validate +
agent retry loop are factored into ``framework/parsers.py`` +
``framework/schemas.py``. Per-shot validation lives on
``ShotSegments.validate_for_shot`` (Pydantic cross-field rules).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from videorlm.framework._common.pools import PLANNER_POOL_SIZE
from videorlm.framework.parsers import ParseError, retry_agent_call, strip_markdown_fence
from videorlm.framework.schemas import BoundaryAnchor, Shot, ShotSegments

from .prompts import SEGMENT_PLANNER_SYSTEM_PROMPT, format_segment_planner_user_prompt

# ``videorlm.backends.llm.agents`` is imported lazily inside functions because
# it transitively requires the vendored ``project`` module on sys.path, which
# the entrypoint (_smoke.py::setup_path) injects only after argparse runs —
# importing here at module-load time crashes during ``from videorlm.framework
# import ...`` before setup_path has a chance to fire.


def make_segment_planner_from_parent_state(
    parent_state: dict[str, Any],
    sp_cfg: Any,
) -> Any:
    """Build an SP agent that swaps system to SEGMENT_PLANNER_SYSTEM_PROMPT
    and inherits the parent's tail (user(story) + assistant(skeleton)).

    Args:
      parent_state: parent agent's state dict; messages[0] is parent's
          system prompt and is dropped during inheritance.
      sp_cfg: QwenConfig for the SP. Its system_prompt field is overridden
          here; caller should still pass system_prompt=SEGMENT_PLANNER_SYSTEM_PROMPT
          for clarity.

    Returns:
      A new QwenAgent, NOT yet started. Caller ``start()``s it (or uses ``with``).
    """
    from videorlm.backends.llm.agents import OpenAIMessagesAgent, QwenAgent
    from videorlm.framework._common.fork import sub_conversation_with_system_swap

    new_state = sub_conversation_with_system_swap(
        parent_state, SEGMENT_PLANNER_SYSTEM_PROMPT,
        inherit_policy="first_pair_only",
    )
    if type(sp_cfg).__name__ == "OpenAIMessagesConfig":
        return OpenAIMessagesAgent(sp_cfg, state=new_state)
    return QwenAgent(sp_cfg, state=new_state)


def plan_segments_for_shot(
    parent_state: dict[str, Any],
    shot: dict[str, Any],
    anchor: dict[str, Any],
    skeleton: dict[str, Any] | None,
    sp_cfg: Any,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    """One shot's segments: build SP from parent_state, prompt once, parse, validate.

    Retries up to ``max_retries`` via ``retry_agent_call`` on JSON parse /
    Pydantic cross-field validation failures (streaming responses
    occasionally truncate). Each retry drops the failed user+assistant
    pair from SP state so history stays clean.

    ``skeleton`` (optional) is forwarded to the user prompt so SP can pick
    portrait/place/prop IDs from a known set without scanning history.
    """
    inherited = len(parent_state.get("messages", []) or []) - 1  # tail length
    print(
        f"[agent-trace] sp for shot={shot['id']} build: inherited_tail={inherited} "
        "(system swapped to SEGMENT_PLANNER_SYSTEM_PROMPT)",
        flush=True,
    )

    # Coerce shot / anchor dicts → Pydantic for cross-field validation. The
    # ShotSegments.validate_for_shot rules need ``Shot.duration_s`` /
    # ``BoundaryAnchor.id`` as model attributes; the dict shape callers
    # pass in is otherwise unchanged.
    shot_pyd = Shot.model_validate(shot)
    anchor_pyd = BoundaryAnchor.model_validate(anchor)

    def _parse(reply: str) -> dict[str, Any]:
        body = strip_markdown_fence(reply)
        raw = json.loads(body)
        # ShotSegments.validate_for_shot raises ValueError / ValidationError
        # on schema breaks — retry_agent_call catches both as Exception.
        ShotSegments.validate_for_shot(raw, shot_pyd, anchor_pyd)
        return raw

    def _on_fail(label: str, attempt: int, exc: BaseException, reply_head: str) -> None:
        print(
            f"[sp shot={shot['id']}] attempt {attempt}/{max_retries} parse/validate failed "
            f"({type(exc).__name__}: {str(exc)[:160]}); reply head: {reply_head!r}",
            flush=True,
        )

    sp = make_segment_planner_from_parent_state(parent_state, sp_cfg)
    user_prompt = format_segment_planner_user_prompt(shot, anchor, planner=skeleton)
    try:
        with sp:
            segs = retry_agent_call(
                sp, user_prompt, _parse,
                max_retries=max_retries,
                sleep_base_s=2.0, sleep_cap_s=60.0,
                on_fail=_on_fail,
                label=f"sp shot={shot['id']}",
            )
            after = len(sp.state.get("messages", []) or [])
            print(
                f"[agent-trace] sp for shot={shot['id']} done: messages={after}",
                flush=True,
            )
            return segs
    except ParseError as exc:
        last = exc.last_exc or exc
        raise RuntimeError(
            f"plan_segments_for_shot[{shot['id']}]: exhausted {max_retries} retries; "
            f"last={type(last).__name__}: {str(last)[:200]}"
        ) from exc


def _validate_segments(segs: dict[str, Any], shot: dict[str, Any], anchor: dict[str, Any]) -> None:
    """Backwards-compat shim: delegates to ``ShotSegments.validate_for_shot``.

    Kept so any ad-hoc smoke scripts
    that still import ``_validate_segments`` keep working without
    changes. Raises ``ValueError`` on schema break (same as legacy API)
    rather than Pydantic's ``ValidationError``.
    """
    from pydantic import ValidationError
    shot_pyd = Shot.model_validate(shot)
    anchor_pyd = BoundaryAnchor.model_validate(anchor)
    try:
        ShotSegments.validate_for_shot(segs, shot_pyd, anchor_pyd)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def plan_segments_all(
    parent_state: dict[str, Any],
    skeleton: dict[str, Any],
    sp_cfg: Any,
    *,
    max_workers: int = PLANNER_POOL_SIZE,
) -> dict[str, Any]:
    """Fan-out: one fresh SP agent per shot, parallel. Each SP forks from
    ``parent_state`` (not from a live parent agent), so the parent doesn't
    have to be alive while shots run.

    The ThreadPoolExecutor upper bound mirrors the framework's
    ``PLANNER_POOL_SIZE`` (env: ``RECA_PLANNER_POOL_SIZE``) so one knob
    controls both the in-thread slot count and the executor's worker
    count — no risk of disagreement between them.
    """
    shots = skeleton["shots"]
    anchors = skeleton["boundarys"]["boundary_anchors"]
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(plan_segments_for_shot, parent_state, shot, anchors[i], skeleton, sp_cfg): shot["id"]
            for i, shot in enumerate(shots)
        }
        for fut in as_completed(futures):
            out.update(fut.result())
    return out


def reconstruct_segment_planner_state(
    planner: dict[str, Any],
    story: str,
    shot_id: str,
) -> dict[str, Any]:
    """Rebuild SP's post-turn state.messages offline.

    Used by validator after the SP agent has been closed (SP runs in a
    thread pool and closes immediately after its turn). Faithful shape:

      [system(SP), user(story), assistant(skeleton_json),
       user(per-shot), assistant(segments_for_shot_json)]

    Args:
      planner: full planner output (skeleton + segments).
      story: original story text (parent's user[0]).
      shot_id: which shot's SP turn to reconstruct.

    Returns:
      A state dict {messages: [AgentMessage, ...]}.
    """
    from videorlm.backends.llm.agents import AgentMessage

    skeleton = {k: v for k, v in planner.items() if k != "segments"}
    skeleton_json = "```json\n" + json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n```"

    shot = next(s for s in planner["shots"] if s["id"] == shot_id)
    shot_idx = next(i for i, s in enumerate(planner["shots"]) if s["id"] == shot_id)
    anchor = planner["boundarys"]["boundary_anchors"][shot_idx]

    sp_user = format_segment_planner_user_prompt(shot, anchor, planner=skeleton)
    segs_for_shot = {sid: s for sid, s in planner["segments"].items() if s["shot_id"] == shot_id}
    segs_json = "```json\n" + json.dumps(segs_for_shot, ensure_ascii=False, indent=2) + "\n```"

    return {
        "messages": [
            AgentMessage(role="system", content=SEGMENT_PLANNER_SYSTEM_PROMPT),
            AgentMessage(role="user", content=story),
            AgentMessage(role="assistant", content=skeleton_json),
            AgentMessage(role="user", content=sp_user),
            AgentMessage(role="assistant", content=segs_json),
        ],
    }


def reconstruct_parent_state(story: str, skeleton: dict[str, Any]) -> dict[str, Any]:
    """Rebuild parent agent's post-skeleton state offline (independent helper).

    Returns state.messages = [system(PARENT_SYSTEM_PROMPT), user(story),
    assistant(skeleton_json)]. Used by replan path on --resume to fork an
    SP-replan agent from an offline-rebuilt parent (live parent agent is
    long gone after planning finished).

    Independent of reconstruct_segment_planner_state (which rebuilds the
    full per-shot SP turn including the assistant(segments) tail).
    """
    from videorlm.backends.llm.agents import AgentMessage
    from videorlm.framework.shot_planner.prompts import PARENT_SYSTEM_PROMPT

    skeleton_json = "```json\n" + json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n```"
    return {
        "messages": [
            AgentMessage(role="system", content=PARENT_SYSTEM_PROMPT),
            AgentMessage(role="user", content=story),
            AgentMessage(role="assistant", content=skeleton_json),
        ],
    }


__all__ = [
    "make_segment_planner_from_parent_state",
    "plan_segments_for_shot",
    "plan_segments_all",
    "reconstruct_parent_state",
    "reconstruct_segment_planner_state",
    "_validate_segments",
]
