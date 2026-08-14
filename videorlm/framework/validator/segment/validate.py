"""Segment-level visual validator + repair dispatch.

Two public entry points:
  - ``run_segment_validator(...)`` — judge a rendered segment's tail PNG.
    Returns a ``SegmentJudgment`` dataclass (9-field).
  - ``apply_segment_repair(judgment, segment_spec)`` — produce an updated
    ``segment_spec`` reflecting the judgment's ``repair_action``
    (RegenerateUnit / RepackPrompt / ReanchorState), or ``None`` if passed.

Failure model (iron rules):
  1. **No backward compat** — old anchor-validator API stays in
     ``validator/anchor/``; this module never imports it.
  2. **No legacy** — no cross-model retry chain. We pick a single
     vision-capable model (``qwen3-vl-plus``, on DashScope) and retry
     within it via the same backoff `retry_until_exhausted` uses
     elsewhere. Text-only models like ``qwen3.6-plus`` cannot judge
     images and would be a silent no-op, so they're not in the chain.
  3. **No cross-backend fallback** — failures of the validator are
     surfaced via ``SegmentValidatorError``; caller decides whether to
     accept the un-validated render (defensive) or to abort.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


from .prompts import (
    SEGMENT_VALIDATOR_SYSTEM_PROMPT,
    format_segment_validator_user_prompt,
)


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

REQUIRED_FIELDS = (
    "segment_id",
    "aesthetic_score",
    "global_alignment_score",
    "action_consistency_score",
    "overall_score",
    "pass_or_not",
    "reason",
    "repair_action",
    "edit_prompt",
)

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Default vision validator model. A single DashScope-hosted Qwen with vision
# support; if it fails, we surface ``SegmentValidatorError``. No fallback
# to a different model / backend (§0 design rule). Callers may pin a
# different model per call via ``run_segment_validator(model=...)``; the
# env var sets the process-wide default.
_VISION_MODEL = os.environ.get("RECA_SEGMENT_VALIDATOR_MODEL", "").strip() or "qwen3-vl-plus"

# Default per-axis pass thresholds. A segment must clear all three axes
# AND clear the overall threshold to count as passed. Both are overridable
# per call (``axis_threshold=`` / ``overall_threshold=``) and process-wide
# via env.
_AXIS_PASS_THRESHOLD = _env_float("RECA_SEGMENT_AXIS_THRESHOLD", 0.6)
_OVERALL_PASS_THRESHOLD = _env_float("RECA_SEGMENT_OVERALL_THRESHOLD", 0.7)

# Default server-side frame sampling rate for the video input.
_VIDEO_SAMPLE_FPS = int(_env_float("RECA_SEGMENT_VALIDATOR_FPS", 10))


RepairAction = Literal["", "RegenerateUnit", "RepackPrompt", "ReanchorState"]


class SegmentValidatorError(RuntimeError):
    """Raised when the segment validator output is malformed or the call fails."""


@dataclass(frozen=True)
class SegmentJudgment:
    """9-field judgment dataclass, frozen for thread-safety."""

    segment_id: str
    aesthetic_score: float
    global_alignment_score: float
    action_consistency_score: float
    overall_score: float
    passed: bool
    reason: str
    repair_action: RepairAction
    edit_prompt: str


def _parse(text: str) -> dict[str, Any]:
    m = _FENCE_RE.search(text)
    payload = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise SegmentValidatorError(
            f"segment validator output not parseable JSON: {e}; raw head: {text[:200]!r}"
        ) from e


def _is_unit_float(x: Any) -> bool:
    return isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0


def _validate_schema(d: dict[str, Any]) -> None:
    missing = set(REQUIRED_FIELDS) - d.keys()
    if missing:
        raise SegmentValidatorError(f"segment validator missing fields: {sorted(missing)}")
    if d["pass_or_not"] not in ("pass", "fail"):
        raise SegmentValidatorError(
            f"pass_or_not must be 'pass' or 'fail', got {d['pass_or_not']!r}"
        )
    for k in (
        "aesthetic_score",
        "global_alignment_score",
        "action_consistency_score",
        "overall_score",
    ):
        if not _is_unit_float(d[k]):
            raise SegmentValidatorError(
                f"{k} must be float in [0.0, 1.0], got {d[k]!r}"
            )
    if d["repair_action"] not in ("", "RegenerateUnit", "RepackPrompt", "ReanchorState"):
        raise SegmentValidatorError(
            f"repair_action must be in (\"\", RegenerateUnit, RepackPrompt, ReanchorState); "
            f"got {d['repair_action']!r}"
        )


def _to_judgment(
    d: dict[str, Any],
    *,
    axis_threshold: float | None = None,
    overall_threshold: float | None = None,
) -> SegmentJudgment:
    aesthetic = float(d["aesthetic_score"])
    global_alignment = float(d["global_alignment_score"])
    action_consistency = float(d["action_consistency_score"])
    overall = float(d["overall_score"])
    # PASS rule: numbers are the single source of truth.
    # The model's declared `pass_or_not` is informational — empirically
    # vision LLMs over-fail (declare "fail" even when their own numerical
    # scores all clear thresholds, because the prompt-asked reason field
    # encourages itemizing minor issues which sound like failures).
    #
    # Per system prompt: pass ⇔ all 3 axes ≥ axis_th AND overall ≥ overall_th.
    # We trust this rule and ignore `d["pass_or_not"]`.
    axis_th = _AXIS_PASS_THRESHOLD if axis_threshold is None else float(axis_threshold)
    overall_th = (
        _OVERALL_PASS_THRESHOLD if overall_threshold is None else float(overall_threshold)
    )
    passed = (
        aesthetic >= axis_th
        and global_alignment >= axis_th
        and action_consistency >= axis_th
        and overall >= overall_th
    )
    return SegmentJudgment(
        segment_id=str(d["segment_id"]),
        aesthetic_score=aesthetic,
        global_alignment_score=global_alignment,
        action_consistency_score=action_consistency,
        overall_score=overall,
        passed=passed,
        reason=str(d["reason"])[:500],
        repair_action=d["repair_action"],  # type: ignore[arg-type]
        edit_prompt=str(d.get("edit_prompt", "")),
    )


def _to_dashscope_video_field(video_url_or_path: str) -> str:
    """Convert a local path or https URL → DashScope native SDK ``video`` field.

    DashScope's OpenAI-compat API rejects ``file://`` URIs but the native
    ``dashscope.MultiModalConversation`` SDK accepts them (per
    https://help.aliyun.com/zh/model-studio/vision — "本地文件作为输入"
    section: file paths are only supported via DashScope Python/Java SDK,
    NOT the OpenAI-compat / HTTP modes). So when the upstream renderer
    didn't publish to OSS (e.g. wan_r2v.py returns ``output_url=None``,
    pipeline falls back to local ``output_path``), we still get the
    validator to see the bytes by prefixing ``file://``.

    https(s):// passes through; other strings are resolved as filesystem
    paths and prefixed with ``file://<absolute-path>``.
    """
    if video_url_or_path.startswith(("http://", "https://")):
        return video_url_or_path
    p = Path(video_url_or_path).resolve()
    if not p.exists():
        raise SegmentValidatorError(
            f"segment video not reachable: not an http(s) URL and local "
            f"file does not exist: {video_url_or_path!r}"
        )
    return f"file://{p}"


def _fork_messages_for_validator(
    parent_agent_state: dict[str, Any],
    user_prompt: str,
    video_field: str,
    portrait_urls: list[str],
    *,
    fps: int = 10,
) -> list[dict[str, Any]]:
    """Build a DashScope-native MultiModal ``messages`` list.

    Inherits the SP's last 4 non-system msgs (story + skeleton +
    per-shot user + assistant segments) — same depth as the legacy
    ``sub_conversation_with_system_swap(inherited_msg_count=4)`` path.

    Final shape:

      [
        {role:"system",    content:[{text: VALIDATOR_SYS}]},
        # inherited tail (text-only)
        {role:"user",      content:[{text: <story>}]},
        {role:"assistant", content:[{text: <skeleton_json>}]},
        {role:"user",      content:[{text: <sp_user>}]},
        {role:"assistant", content:[{text: <segments_json>}]},
        # current turn (video primary, portraits as identity refs, then text)
        {role:"user", content:[
          {video: file://... or https://..., fps: 10},
          {image: <portrait_url>}, ...
          {text: <user_prompt>},
        ]},
      ]
    """
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": [{"text": SEGMENT_VALIDATOR_SYSTEM_PROMPT}]},
    ]
    history = list(parent_agent_state.get("messages", []) or [])
    non_sys = [m for m in history if getattr(m, "role", "") != "system"]
    tail = non_sys[-4:]
    for m in tail:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", "") or ""
        msgs.append({"role": role, "content": [{"text": content}]})
    current: list[dict[str, Any]] = [{"video": video_field, "fps": fps}]
    for url in portrait_urls or []:
        if url:
            current.append({"image": url})
    current.append({"text": user_prompt})
    msgs.append({"role": "user", "content": current})
    return msgs


def _call_qwen_vl_native(
    messages: list[dict[str, Any]],
    *,
    model: str,
    timeout_s: float,
) -> str:
    """One-shot DashScope native ``MultiModalConversation.call``.

    Differences vs the OpenAI-compat path the rest of the agent stack
    uses (see ``backends/llm/agents/openai_compat/agent.py``):

      - **local ``file://`` paths are supported** for ``video``/``image``
        fields (the whole reason this exists; the OpenAI-compat endpoint
        rejects ``file://`` with a 400).
      - non-streaming. The validator response is small (~300 tokens) and
        the request goes straight to ``dashscope.aliyuncs.com`` with no
        Cloudflare in front, so the 524 origin-timeout that forces every
        other path to stream isn't a concern.
      - multi-key rotation goes through the shared ``with_key("dashscope", ...)``
        middleware so cooldown / EWMA stats stay consistent with the rest
        of the stack.
      - per-call concurrency is gated by the same ``slot`` semaphore the
        Agent stack uses (key = ``qwen3-vl-plus``, size =
        ``SEGMENT_VALIDATOR_POOL_SIZE``) so we don't accidentally
        flood TPS when ~18 shots validate in parallel.
    """
    from dashscope import MultiModalConversation

    from videorlm.backends._common.platforms import with_key
    from videorlm.backends.llm.agents.base import slot
    from videorlm.framework._common.pools import SEGMENT_VALIDATOR_POOL_SIZE

    def _do(api_key: str):
        return MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            timeout=int(timeout_s),
        )

    with slot("qwen3-vl-plus", SEGMENT_VALIDATOR_POOL_SIZE):
        try:
            resp = with_key("dashscope", _do)
        except BaseException as exc:
            raise SegmentValidatorError(
                f"DashScope MultiModal call failed (model={model!r}): "
                f"{type(exc).__name__}: {str(exc)[:1500]}"
            ) from exc

    status = getattr(resp, "status_code", None)
    if status is not None and status != 200:
        raise SegmentValidatorError(
            f"DashScope MultiModal returned status={status}: "
            f"code={getattr(resp, 'code', '?')!r} "
            f"message={str(getattr(resp, 'message', ''))[:300]!r}"
        )
    try:
        choice = resp.output.choices[0]
        content = choice.message.content
    except Exception as exc:
        raise SegmentValidatorError(
            f"DashScope MultiModal response shape unexpected: "
            f"{type(exc).__name__}: {str(exc)[:300]}; raw_head={str(resp)[:500]!r}"
        ) from exc
    if isinstance(content, list):
        text_chunks: list[str] = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                text_chunks.append(str(c["text"]))
            elif isinstance(c, str):
                text_chunks.append(c)
        reply = "".join(text_chunks)
    elif isinstance(content, str):
        reply = content
    else:
        reply = ""
    if not reply:
        raise SegmentValidatorError(
            f"DashScope MultiModal returned empty content; raw_head={str(resp)[:500]!r}"
        )
    return reply


def run_segment_validator(
    segment_spec: dict[str, Any],
    shot_start_state: str,
    segment_mp4_url: str,
    portrait_urls: list[str],
    parent_agent_state: dict[str, Any],
    *,
    shot_story_goal: str = "",
    shot_visual_intent: str = "",
    model: str | None = None,
    axis_threshold: float | None = None,
    overall_threshold: float | None = None,
    fps: int | None = None,
) -> SegmentJudgment:
    """Judge a rendered segment's full mp4 against its spec (3 axes).

    The validator sees the **complete video timeline** via DashScope
    native ``MultiModalConversation`` (qwen3-vl-plus samples its own
    frames over the timeline), NOT a single extracted tail PNG. This
    lets the action_consistency axis judge physical-contact continuity,
    motif preservation across frames, and identity stability over time —
    none of which a single frame can answer.

    ``segment_mp4_url`` may be either:
      - an http(s):// OSS URL (happyhorse path, DashScope returns one
        from its async-job result), or
      - a local filesystem path (wan path, since wan's
        ``_primitives.py::generate_r2v_video`` discards the
        ``d["video_url"]`` and only saves locally — see
        ``backends/media/impl/dashscope/video/_primitives.py:102``).

    Local paths are converted to ``file://`` and passed through DashScope
    native SDK, which is the **only** mode that accepts ``file://``
    (OpenAI-compat / HTTP modes reject it — confirmed in the official
    docs).

    Args:
      segment_spec: ``render_plan["segments"][seg_id]`` entry — must contain
        ``segment_request`` with ``prompt``, plus top-level ``id`` and
        ``end_state``.
      shot_start_state: parent shot's ``start_state`` (motif source).
      segment_mp4_url: OSS URL OR local mp4 path. See above.
      portrait_urls: identity refs (may be empty).
      parent_agent_state: SP state used to inherit the 4-msg conversation
        tail (story + skeleton + sp_user + segments_json).
      shot_story_goal: parent shot's ``story_goal`` (global_alignment axis).
      shot_visual_intent: parent shot's ``visual_intent`` (global +
        aesthetic axes).

    Returns:
      A ``SegmentJudgment`` (frozen dataclass with per-axis scores +
      ``passed`` boolean computed from thresholds, not the model's
      ``pass_or_not`` self-declaration).

    Raises:
      ``SegmentValidatorError`` if the model output is malformed or the
      API call fails.
    """
    sr = segment_spec.get("segment_request") or {}
    segment_prompt = sr.get("prompt", "")
    segment_id = segment_spec.get("id") or sr.get("request_id") or "?"
    segment_end_state = segment_spec.get("end_state") or ""

    video_field = _to_dashscope_video_field(segment_mp4_url)

    user_prompt = format_segment_validator_user_prompt(
        segment_id=segment_id,
        segment_prompt=segment_prompt,
        segment_end_state=segment_end_state,
        shot_start_state=shot_start_state,
        segment_video_url=segment_mp4_url,
        portrait_urls=portrait_urls,
        shot_story_goal=shot_story_goal,
        shot_visual_intent=shot_visual_intent,
    )

    portraits = [u for u in (portrait_urls or []) if u]
    messages = _fork_messages_for_validator(
        parent_agent_state, user_prompt, video_field, portraits,
        fps=int(fps) if fps else _VIDEO_SAMPLE_FPS,
    )

    model_name = (model or "").strip() or _VISION_MODEL
    timeout_s = float(os.environ.get("M12_SEGMENT_VALIDATOR_TIMEOUT_S", "900"))
    try:
        reply = _call_qwen_vl_native(messages, model=model_name, timeout_s=timeout_s)
    except SegmentValidatorError as exc:
        raise SegmentValidatorError(
            f"segment validator {model_name} call failed for {segment_id}: "
            f"{str(exc)[:1500]}"
        ) from exc

    d = _parse(reply)
    _validate_schema(d)
    return _to_judgment(
        d, axis_threshold=axis_threshold, overall_threshold=overall_threshold,
    )


def _weakest_axis(j: SegmentJudgment) -> str:
    axes = {
        "aesthetic": j.aesthetic_score,
        "global_alignment": j.global_alignment_score,
        "action_consistency": j.action_consistency_score,
    }
    return min(axes, key=axes.get)  # type: ignore[arg-type]


def apply_segment_repair(
    judgment: SegmentJudgment,
    segment_spec: dict[str, Any],
) -> dict[str, Any] | None:
    """Produce an updated ``segment_spec`` reflecting the judgment's repair
    action. Returns ``None`` if the judgment passed (no repair needed).

    The mutation is always a deep copy — caller's original spec is
    untouched so history is preserved for traceability.

    Action semantics (per ``prompts.py`` decision table):
      - ``RegenerateUnit``: same prompt, re-render the same unit. (Identity
        between calls is up to the backend's seed handling — same seed →
        same output, so callers should bump seed or accept identical
        result and rely on stochastic differences if any.)
      - ``RepackPrompt``: append ``edit_prompt`` ("[强化]: ...") to the
        segment.prompt so the model gets an extra clause for the weakest
        axis (usually motif / identity / story_goal anchor).
      - ``ReanchorState``: treated like RegenerateUnit for now — true
        re-anchor would mean dropping back to the anchor stage, which is
        out of scope for a single-shot repair.
    """
    if judgment.passed:
        return None

    if judgment.repair_action == "RepackPrompt":
        new_spec = copy.deepcopy(segment_spec)
        sr = new_spec.setdefault("segment_request", {})
        old_prompt = sr.get("prompt", "")
        edit = (judgment.edit_prompt or "").strip()
        if not edit:
            # No specific repair guidance — degrade to RegenerateUnit.
            return new_spec
        new_prompt = old_prompt + "\n[强化]: " + edit
        sr["prompt"] = new_prompt
        return new_spec

    if judgment.repair_action in ("RegenerateUnit", "ReanchorState"):
        return copy.deepcopy(segment_spec)

    # Empty action ⇒ no repair attempted.
    return None


__all__ = [
    "SEGMENT_VALIDATOR_SYSTEM_PROMPT",
    "SegmentJudgment",
    "SegmentValidatorError",
    "REQUIRED_FIELDS",
    "apply_segment_repair",
    "run_segment_validator",
]
