"""Sub-conversation helper for stateful agents.

Single source of truth for "clone an agent state + swap system prompt".

Replaces the 5 hardcoded `[1:]` tail-inheritance sites:

  - segment_planner/plan.py        (main SP fork — initial)
  - segment_planner/plan.py        (SP replan fork)
  - validator/anchor/validate.py   (anchor validator fork from SP)
  - validator/segment/validate.py  (segment validator, inherited_msg_count=4)

Rationale (see `docs/framework.md §2.2`): naive `[1:]` inheritance blows the
token budget when long shot counts produce 35+ history messages. Most
callers only need the first ``[user(story), assistant(skeleton)]`` pair,
not the whole history.

Renamed from ``fork_with_system_swap`` to clarify semantics — this is NOT
a process / thread fork (no parallel execution); it materializes a new
conversation seeded from a prefix of the parent's history with a fresh
system prompt. The new state always gets a fresh ``thread_id`` so
underlying transports (codex / qwen / etc.) treat it as a brand-new
conversation.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Literal


InheritPolicy = Literal["all", "first_pair_only", "last_n_pairs", "none"]


def _new_thread_id() -> str:
    return f"sub-{uuid.uuid4().hex[:12]}"


def _last_n_pairs(msgs: list[Any], n_pairs: int) -> list[Any]:
    """Return the trailing ``n_pairs`` (user, assistant) tuples from ``msgs``,
    plus any dangling orphan messages at the very end.

    ``msgs`` is expected to start with ``[system, ...]``; the system message
    is skipped. The scan walks BACKWARD from the tail so:

      - Orphan trailing messages (e.g. a dangling user with no assistant
        reply yet, or a tool-call burst) are preserved at the end.
      - Pair order is preserved in the output.

    This is the "head-scanning" `_last_n_pairs` from the original framework
    version, REPLACED by the tail-backward scan from the deleted
    ``backends/llm/agents/_fork.py``, which is correct on more edge cases.
    """
    body = list(msgs[1:]) if msgs and getattr(msgs[0], "role", None) == "system" else list(msgs)
    if n_pairs <= 0 or not body:
        return []

    pairs: list[tuple[Any, Any]] = []
    trailing: list[Any] = []
    i = len(body) - 1
    # Gather trailing non-assistant suffix into `trailing` first.
    while i >= 0 and getattr(body[i], "role", None) != "assistant":
        trailing.insert(0, body[i])
        i -= 1
    # Walk back over (user → assistant) pairs.
    while i >= 1 and len(pairs) < n_pairs:
        if (
            getattr(body[i], "role", None) == "assistant"
            and getattr(body[i - 1], "role", None) == "user"
        ):
            pairs.insert(0, (body[i - 1], body[i]))
            i -= 2
        else:
            i -= 1

    out: list[Any] = []
    for u, a in pairs:
        out.append(u)
        out.append(a)
    out.extend(trailing)
    return out


def sub_conversation_with_system_swap(
    parent_state: dict[str, Any],
    new_system: str,
    *,
    inherited_msg_count: int | None = None,
    inherit_policy: InheritPolicy = "first_pair_only",
) -> dict[str, Any]:
    """Build a new agent state from ``parent_state`` with a fresh system
    prompt and a sliced subset of the parent's history.

    Parameters
    ----------
    parent_state:
        Dict with ``"messages"`` key (list of ``AgentMessage``-like). The
        original is not mutated.
    new_system:
        Replacement system prompt for the new state.
    inherited_msg_count:
        If not None, take the LAST ``n`` messages of the parent's non-system
        tail (``msgs[1:][-n:]``). Overrides ``inherit_policy``. 0 means
        inherit none.
    inherit_policy (when ``inherited_msg_count`` is None):
      - ``"all"``             — inherit everything except parent's system
                                (legacy ``[1:]`` behavior)
      - ``"first_pair_only"`` — keep only the first
                                ``[user(story), assistant(skeleton)]`` pair.
                                Default. Best for SP forks where token
                                budget matters more than full history.
      - ``"last_n_pairs"``    — last ``RECA_FORK_INHERIT_PAIRS`` pairs
                                (default 2). Useful when later turns
                                carry more context than earlier ones.
      - ``"none"``            — fresh chat, no tail inheritance.

    Returns
    -------
    A new state ``{"messages": [system(new), ...tail], "thread_id": ...}``.
    The new ``thread_id`` is always freshly generated (``"sub-<uuid>"``)
    so transports treat it as a brand-new conversation.
    """
    # Lazy import — framework callers shouldn't pay the backends import cost
    # just to load this helper.
    from videorlm.backends.llm.agents import AgentMessage

    msgs = list(parent_state.get("messages", []) or [])

    if inherited_msg_count is not None:
        if inherited_msg_count <= 0:
            tail = []
        else:
            non_sys = msgs[1:]
            tail = list(non_sys[-inherited_msg_count:])
    elif inherit_policy == "all":
        tail = msgs[1:]
    elif inherit_policy == "first_pair_only":
        tail = msgs[1:3] if len(msgs) >= 3 else msgs[1:]
    elif inherit_policy == "last_n_pairs":
        n_pairs = 2
        raw = os.environ.get("RECA_FORK_INHERIT_PAIRS", "").strip()
        if raw:
            try:
                n_pairs = max(0, int(raw))
            except ValueError:
                pass
        tail = _last_n_pairs(msgs, n_pairs)
    elif inherit_policy == "none":
        tail = []
    else:
        raise ValueError(f"unknown inherit_policy: {inherit_policy!r}")

    return {
        "thread_id": _new_thread_id(),
        "messages": [AgentMessage(role="system", content=new_system), *tail],
    }


__all__ = ["sub_conversation_with_system_swap", "InheritPolicy"]
