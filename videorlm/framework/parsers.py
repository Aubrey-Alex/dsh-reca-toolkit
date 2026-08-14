"""Unified output parser for planner / replanner / validator JSON replies.

Replaces three near-identical copies of ``_parse_json_block`` +
``_validate_*`` scattered across:
  - framework/shot_planner/plan.py
  - framework/segment_planner/plan.py

Inspired by langchain's ``PydanticOutputParser`` + ``OutputFixingParser``:
  1. Generic over a Pydantic model class.
  2. ``format_instructions()`` produces a stable JSON-schema blurb to be
     injected into the user prompt — the LLM knows exactly what shape
     to emit.
  3. ``parse(raw)`` strips markdown fences and runs ``model_validate_json``.
  4. ``parse_with_retry(agent, prompt, ...)`` does the prompt → reply →
     parse → on-fail retry loop in one place; callers don't reinvent
     the drop-last-pair-from-history dance.

Usage:
    skeleton_parser = JSONOutputParser(Skeleton)
    skel = skeleton_parser.parse_with_retry(parent, story_prompt, max_retries=30)

The ``Callback`` integration (parsers fire on_parse_fail / on_llm_call)
is wired in stage.py; here we only need a thin optional callback hook so
parsers used in standalone scripts still work without a full RunContext.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(\{.*\})\s*```", re.DOTALL)


def strip_markdown_fence(text: str) -> str:
    """Pull a JSON body out of optional ```json ... ``` markdown fences.

    Mirrors the three legacy ``_parse_json_block`` regexes but exposed as
    a standalone utility so non-Pydantic callers can reuse it.
    """
    m = _FENCE_RE.search(text)
    return (m.group(1) if m else text).strip()


class ParseError(Exception):
    """Raised by ``JSONOutputParser`` when the LLM reply can't be coerced
    into the target schema after retries. Holds the last reply for debug.
    """

    def __init__(self, msg: str, *, reply_head: str = "", last_exc: Exception | None = None):
        super().__init__(msg)
        self.reply_head = reply_head
        self.last_exc = last_exc


# Optional callback hook: parsers don't need the full Callback protocol;
# a single fn(stage, attempt, error, reply_head) is enough. stage.py
# bridges this to the full Callback list when wired up.
OnParseFail = Callable[[str, int, Exception, str], None]


# ─── generic retry helper (used by JSONOutputParser + plan helpers) ───────


def retry_agent_call(
    agent: Any,
    prompt: str,
    parse: Callable[[str], Any],
    *,
    max_retries: int = 10,
    sleep_base_s: float = 2.0,
    sleep_cap_s: float = 60.0,
    on_fail: OnParseFail | None = None,
    label: str = "retry_agent_call",
) -> Any:
    """Drive ``agent.prompt(prompt) → parse(reply)`` with a uniform retry loop.

    Unifies the three near-identical retry loops in shot_planner /
    segment_planner. ``parse(reply)`` is any
    function that raises on failure (JSON syntax, schema mismatch,
    cross-field invariant violation) and returns the parsed value on
    success. The loop handles:

      - ``agent.prompt(...)`` exceptions (Cloudflare 524 / origin
        timeout / transient gateway errors): drop the trailing user
        message from history so retry starts clean, back off, retry.
      - ``parse(reply)`` exceptions: drop the failed user+assistant
        pair from history so retry isn't biased by garbage, back off,
        retry.

    The exact "drop tail" surgery is what every legacy retry loop did
    by hand; centralising it here is the point of this helper.

    Args:
      agent: any object with ``state.messages`` (a list with ``role``
        attribute) and a ``prompt(text) -> str`` method.
      prompt: user message body to send each attempt.
      parse: parse-and-validate callable; raises on failure.
      max_retries: total attempts before raising ``ParseError``.
      sleep_base_s, sleep_cap_s: exponential backoff between attempts.
      on_fail: optional callback fired after each failed attempt
        ``(label, attempt, exc, reply_head)``. Used by stage.py to
        forward into the full ``Callback`` chain.
      label: short name for log messages (e.g. ``"plan_skeleton"`` or
        ``"sp shot=shot07_man_wait"``).
    """
    last_exc: Exception | None = None
    last_reply_head = ""
    for attempt in range(max_retries):
        try:
            reply = agent.prompt(prompt)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if on_fail is not None:
                on_fail(label, attempt + 1, exc, "")
            msgs = list(getattr(agent.state, "get", lambda *_: [])("messages", []) or [])
            if msgs and getattr(msgs[-1], "role", None) == "user":
                agent.state["messages"] = msgs[:-1]
            if attempt < max_retries - 1:
                time.sleep(min(sleep_cap_s, sleep_base_s * (attempt + 1)))
            continue

        try:
            return parse(reply)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            last_reply_head = reply[:200]
            if on_fail is not None:
                on_fail(label, attempt + 1, exc, last_reply_head)
            msgs = list(getattr(agent.state, "get", lambda *_: [])("messages", []) or [])
            if (
                len(msgs) >= 2
                and getattr(msgs[-1], "role", None) == "assistant"
                and getattr(msgs[-2], "role", None) == "user"
            ):
                agent.state["messages"] = msgs[:-2]
            if attempt < max_retries - 1:
                time.sleep(min(sleep_cap_s, sleep_base_s * (attempt + 1)))

    raise ParseError(
        f"{label}: exhausted {max_retries} retries; "
        f"last={type(last_exc).__name__}: {str(last_exc)[:200] if last_exc else 'unknown'}",
        reply_head=last_reply_head,
        last_exc=last_exc,
    )


class JSONOutputParser(Generic[T]):
    """Schema-driven JSON parser for one Pydantic model class.

    Stateless and cheap to construct — keep one instance per stage as a
    module-level singleton:

        SkeletonParser = JSONOutputParser(Skeleton)
        ShotSegmentsParser = JSONOutputParser(ShotSegments)
    """

    def __init__(self, cls: type[T], *, name: str | None = None):
        self.cls = cls
        self.name = name or cls.__name__

    # ── format_instructions: inject into user prompt ────────────────────

    def format_instructions(self, *, include_schema: bool = True) -> str:
        """Return a Chinese-friendly blurb that tells the LLM the exact
        JSON schema it must emit. Designed to be appended verbatim to the
        user message (after the natural-language task description)."""
        if not include_schema:
            return "请用一个 JSON 对象作为回复,不要有额外文字。"
        schema = json.dumps(self.cls.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            f"请用一个 JSON 对象作为回复,严格匹配以下 schema(允许多余字段,"
            f"但 required 字段不可缺失):\n\n```json\n{schema}\n```\n\n"
            "回复以 `{` 开始,以 `}` 结束,可以包裹在 ```json ... ``` 中。"
        )

    # ── parse: one-shot ─────────────────────────────────────────────────

    def parse(self, raw: str) -> T:
        """Parse a raw LLM reply into ``self.cls``. Raises ``ParseError``
        on JSON syntax / schema mismatch with the reply head attached."""
        body = strip_markdown_fence(raw)
        try:
            return self.cls.model_validate_json(body)
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
            raise ParseError(
                f"{self.name}.parse failed: {type(exc).__name__}: {str(exc)[:200]}",
                reply_head=raw[:200],
                last_exc=exc,
            ) from exc

    # ── parse_with_retry: drives the agent in a loop ────────────────────

    def parse_with_retry(
        self,
        agent: Any,
        prompt: str,
        *,
        max_retries: int = 10,
        sleep_base_s: float = 2.0,
        sleep_cap_s: float = 60.0,
        on_parse_fail: OnParseFail | None = None,
    ) -> T:
        """Prompt → reply → ``self.parse(reply)`` → on-failure retry.

        Thin wrapper over ``retry_agent_call`` that fixes the parse
        function to ``self.parse``. See ``retry_agent_call`` for the
        full behaviour contract.
        """
        return retry_agent_call(
            agent,
            prompt,
            self.parse,
            max_retries=max_retries,
            sleep_base_s=sleep_base_s,
            sleep_cap_s=sleep_cap_s,
            on_fail=on_parse_fail,
            label=self.name,
        )


__all__ = [
    "JSONOutputParser",
    "ParseError",
    "OnParseFail",
    "strip_markdown_fence",
    "retry_agent_call",
]
