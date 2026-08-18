"""Validator functions: anchor visual judgment + image-edit repair loop.

Pipeline integration: after `_dispatch_one_image` succeeds for an anchor,
call `validate_anchor()`. On `pass_or_not == "fail"`, call
`apply_image_edit_repair()` to regenerate via image-edit backend, then
re-validate (or accept the last attempt after `max_repair_attempts`).

The two backend touchpoints (`vision_call` and `image_edit_call`) are
passed in as plain callables so this module stays decoupled from
`backends.llm` / `backends.media` choices — caller wires them at the
pipeline glue level.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .prompts import VALIDATOR_SYSTEM_PROMPT, format_validator_user_prompt


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

REQUIRED_FIELDS = (
    "anchor_id", "score", "reason", "pass_or_not",
    "reference_inputs", "negative_prompt", "edit_prompt",
)


class ValidatorError(RuntimeError):
    """Raised when validator output is malformed (parse / schema)."""


def _parse_validator_output(text: str) -> dict[str, Any]:
    """Extract the 7-field JSON from a vision-LLM reply.

    Tolerates ```json fence and bare JSON. Raises ValidatorError on
    unparseable input.
    """
    m = _FENCE_RE.search(text)
    payload = m.group(1).strip() if m else text.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise ValidatorError(f"validator output not parseable JSON: {e}; raw: {text[:200]!r}") from e


def _validate_schema(d: dict[str, Any]) -> None:
    """Hard-check the 7 required fields are present + pass_or_not is canonical."""
    missing = set(REQUIRED_FIELDS) - d.keys()
    if missing:
        raise ValidatorError(f"validator output missing fields: {missing}")
    if d["pass_or_not"] not in ("pass", "fail"):
        raise ValidatorError(f"pass_or_not must be 'pass' or 'fail', got {d['pass_or_not']!r}")
    if not isinstance(d["score"], (int, float)) or not (0.0 <= float(d["score"]) <= 1.0):
        raise ValidatorError(f"score must be float in [0.0, 1.0], got {d['score']!r}")
    ri = d["reference_inputs"]
    if not isinstance(ri, dict) or "origin" not in ri:
        raise ValidatorError(f"reference_inputs must be dict with 'origin' key, got {ri!r}")


def validate_anchor(
    *,
    anchor_id: str,
    shot_id: str,
    anchor_prompt: str,
    expected_start_state: str,
    anchor_image_url: str,
    reference_inputs: dict[str, Any],
    vision_call: Callable[[str, str, str], str],
) -> dict[str, Any]:
    """Send anchor PNG + spec to vision LLM, return 7-field judgment dict.

    Args:
      anchor_id, shot_id: identifiers carried into the JSON output.
      anchor_prompt: original anchor.prompt the renderer was given.
      expected_start_state: shot.start_state for cross-check.
      anchor_image_url: OSS / public URL of the rendered PNG.
      reference_inputs: {"portrait": "...", "place": "...", "prop": "..."} ID strings.
      vision_call: pluggable callable
          (system_prompt: str, user_prompt: str, image_url: str) -> str
          Implementor wires from `videorlm.backends.llm.vision_completion`
          or equivalent. Should return the raw text reply (may contain
          ```json fence) which we then parse.

    Returns:
      dict matching videorlm/good_examples/validator_example.json schema
      (7 strict fields).

    Raises:
      ValidatorError: when LLM reply is unparseable or schema-invalid.
    """
    user_prompt = format_validator_user_prompt(
        anchor_id, shot_id, anchor_prompt, expected_start_state,
        anchor_image_url, reference_inputs,
    )
    raw_reply = vision_call(VALIDATOR_SYSTEM_PROMPT, user_prompt, anchor_image_url)
    judgment = _parse_validator_output(raw_reply)
    _validate_schema(judgment)
    return judgment


def apply_image_edit_repair(
    *,
    judgment: dict[str, Any],
    source_image_url: str,
    output_path: str | Path,
    image_edit_call: Callable[..., str],
    extra_reference_urls: list[str] | None = None,
) -> str:
    """Re-render the anchor via image-edit using validator's edit_prompt.

    Reference list ordering convention (chain-critical):
      references = [source_image_url, *extra_reference_urls]
                    ↑ index 0: ALWAYS the failed anchor PNG to be edited
                    ↑ index 1+: original portrait / place / prop URLs
                                used to preserve identity during edit

    The first slot is the image-edit backend's "image to modify"; the
    rest are visual references the model should respect for identity
    consistency (so the edit doesn't drift the character / scene). This
    matches what gpt-image-2-pro's image-edit endpoint expects.

    Args:
      judgment: validator output dict; must have pass_or_not="fail".
      source_image_url: the failed anchor PNG URL (will be index 0 of refs).
      output_path: where the repaired PNG should land locally.
      image_edit_call: pluggable callable
          (prompt, source_url, references, output_path, negative_prompt) -> str
      extra_reference_urls: optional portrait/place/prop URLs to preserve.
          Caller should resolve these from the original render_plan's
          anchor reference_inputs[].asset_id → asset_pool[asset_id].

    Returns:
      URL of the new (edited) anchor PNG.

    Raises:
      ValueError: if judgment is not a "fail" verdict.
    """
    if judgment.get("pass_or_not") != "fail":
        raise ValueError(
            f"apply_image_edit_repair only valid on pass_or_not='fail', got {judgment.get('pass_or_not')!r}"
        )
    refs = [source_image_url] + list(extra_reference_urls or [])
    return image_edit_call(
        prompt=judgment["edit_prompt"],
        source_url=source_image_url,
        references=refs,
        output_path=str(output_path),
        negative_prompt=judgment.get("negative_prompt", ""),
    )


def validate_and_repair(
    *,
    anchor_id: str,
    shot_id: str,
    anchor_prompt: str,
    expected_start_state: str,
    anchor_image_url: str,
    output_path: str | Path,
    reference_inputs: dict[str, Any],
    vision_call: Callable[[str, str, str], str],
    image_edit_call: Callable[..., str],
    max_repair_attempts: int = 2,
    extra_reference_urls: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate; if fail, repair via image-edit, re-validate, until pass or max.

    Returns:
      (final_image_url, judgment_history)
        - final_image_url: the URL accepted (could be original or repaired)
        - judgment_history: list of validator outputs in order

    The loop accepts the last image even if still "fail" after
    `max_repair_attempts` (caller can decide what to do with stuck cases
    via inspecting `history[-1]`).
    """
    history: list[dict[str, Any]] = []
    current_url = anchor_image_url
    for attempt in range(max_repair_attempts + 1):
        judgment = validate_anchor(
            anchor_id=anchor_id, shot_id=shot_id,
            anchor_prompt=anchor_prompt,
            expected_start_state=expected_start_state,
            anchor_image_url=current_url,
            reference_inputs=reference_inputs,
            vision_call=vision_call,
        )
        history.append(judgment)
        if judgment["pass_or_not"] == "pass":
            print(
                f"[validator] {anchor_id} attempt={attempt+1} score={judgment['score']:.2f} → pass",
                flush=True,
            )
            return current_url, history
        if attempt >= max_repair_attempts:
            print(
                f"[validator] {anchor_id} fail after {attempt+1} attempts (last score={judgment['score']:.2f}), accepting last",
                flush=True,
            )
            return current_url, history
        repair_path = Path(output_path).with_suffix(f".repair_{attempt+1}.png")
        print(
            f"[validator] {anchor_id} attempt={attempt+1} score={judgment['score']:.2f} → fail, image-edit repair → {repair_path.name}",
            flush=True,
        )
        current_url = apply_image_edit_repair(
            judgment=judgment, source_image_url=current_url,
            output_path=repair_path, image_edit_call=image_edit_call,
            extra_reference_urls=extra_reference_urls,
        )
    return current_url, history


def make_validator_from_segment_planner_state(
    segment_planner_state: dict[str, Any],
    validator_config: Any,
) -> Any:
    """Option B: build a validator agent that inherits segment_planner messages
    BUT replaces system_prompt with VALIDATOR_SYSTEM_PROMPT.

    The new agent's state.messages = [
        AgentMessage(role="system", content=VALIDATOR_SYSTEM_PROMPT),  # NEW system
        ...segment_planner.state.messages[1:]  # inherited tail (story + skeleton + segment task + segments)
    ]

    So the validator can reference the full planning context (it knows what
    the shot is for, how segments are split, what the anchor spec was) but
    speaks with the validator's role.

    Args:
      segment_planner_state: dict with "messages" key (list[AgentMessage]).
          Typically obtained from `segment_planner.state` after
          `parent.fork()` + plan_segments_for_shot turn completed.
      validator_config: a QwenConfig (or compatible) with model + api_key
          + base_url set. system_prompt field is overridden by this
          factory; caller should pass system_prompt=VALIDATOR_SYSTEM_PROMPT
          for clarity but it doesn't affect the new state.

    Returns:
      A new QwenAgent instance, NOT yet started. Caller should `start()`
      it (or use `with` block) before calling `prompt_with_images()`.
    """
    from videorlm.backends.llm.agents import OpenAIResponsesAgent, QwenAgent
    from videorlm.framework._common.fork import sub_conversation_with_system_swap
    from .prompts import VALIDATOR_SYSTEM_PROMPT

    # The full segment-planner transcript is useful for offline diagnosis but
    # makes Responses requests unnecessarily large and slow on gateways with
    # strict context/proxy limits. Production Director audits only need the
    # current anchor prompt and state, which are supplied in the next user
    # turn. Opt into the compact request shape for that path.
    if os.environ.get("RECA_AUDIT_MINIMAL_CONTEXT", "0") == "1":
        new_state = {"messages": []}
    else:
        new_state = sub_conversation_with_system_swap(
            segment_planner_state, VALIDATOR_SYSTEM_PROMPT,
            inherited_msg_count=4,
        )
    if getattr(validator_config, "provider", "") == "openai_responses":
        return OpenAIResponsesAgent(validator_config, state=new_state)
    return QwenAgent(validator_config, state=new_state)


def validate_anchor_via_agent(
    *,
    validator_agent: Any,
    anchor_id: str,
    shot_id: str,
    anchor_prompt: str,
    expected_start_state: str,
    anchor_image_url: str,
    reference_inputs: dict[str, Any],
    expected_end_state: str = "",
    portrait_refs: list[dict[str, Any]] | None = None,
    prop_refs: list[dict[str, Any]] | None = None,
    place_refs: list[dict[str, Any]] | None = None,
    reference_portrait_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Validate using a pre-configured validator agent (Option B path).

    The agent is expected to already carry:
      - VALIDATOR_SYSTEM_PROMPT as messages[0] (system role)
      - inherited segment_planner conversation tail (story + skeleton +
        per-shot segment task + segments output) as messages[1..N]

    This skips the `vision_call` indirection of `validate_anchor()` —
    the agent IS the vision call, and per-call setup (system + history)
    is already baked into its state via `make_validator_from_segment_planner_state`.

    Args:
      validator_agent: a started agent with `prompt_with_images()` method.
      anchor_id, shot_id, anchor_prompt, expected_start_state,
      anchor_image_url, reference_inputs: same as `validate_anchor()`,
      used to build the per-anchor user prompt.
      expected_end_state: shot.end_state, included so validator can
          disambiguate T=0 vs T>0.
      portrait_refs, prop_refs, place_refs: optional lists of dicts, each
          ``{id, name, prompt}`` for the assets referenced by this
          anchor. Rendered as TEXT description blocks in the user prompt
          (no images on the vision channel). The validator reads the
          design prompt of each ref to know what variant SHOULD appear
          in the anchor — enough for variant disambiguation
          (``sun_wukong`` vs ``giant_ape_wukong``) without the latency
          and attention cost of inlining multiple PNGs.
      reference_portrait_urls: DEPRECATED. Kept for backward compat with
          the multi-image path; if non-empty the URLs are still sent on
          the vision channel after the anchor. Pipeline.py now passes
          only ``portrait_refs`` text descriptions.

    Returns:
      7-field judgment dict matching validator_example.json schema.

    Raises:
      ValidatorError on parse / schema failure.
    """
    from .prompts import format_validator_user_prompt

    user_prompt = format_validator_user_prompt(
        anchor_id, shot_id, anchor_prompt, expected_start_state,
        anchor_image_url, reference_inputs,
        expected_end_state=expected_end_state,
        portrait_refs=portrait_refs,
        prop_refs=prop_refs,
        place_refs=place_refs,
    )
    image_urls = [anchor_image_url, *(reference_portrait_urls or [])]
    raw_reply = validator_agent.prompt_with_images(user_prompt, image_urls)
    judgment = _parse_validator_output(raw_reply)
    _validate_schema(judgment)
    return judgment


__all__ = [
    "ValidatorError",
    "validate_anchor",
    "validate_anchor_via_agent",
    "make_validator_from_segment_planner_state",
    "apply_image_edit_repair",
    "validate_and_repair",
    "REQUIRED_FIELDS",
]
