"""DashScope provider — Aliyun Model Studio (wan / wanx / qwen-image
async tasks + qwen LLM chat).

Rate limits source: ``configs/dashscope_rate_limits.json`` (official
Aliyun numbers, see file _meta block for fetch date + URL).
Response classifier: shared with media + LLM dashscope call sites via
``classify_dashscope_response``.
"""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from ..key_pool import classify_error
from . import register_provider
from ._base import Provider


def _repo_configs_dir() -> Path:
    """``unirlm-02/configs/`` relative to this file.

    parents[0] = providers/, [1] = _common/, [2] = backends/,
    [3] = videorlm/, [4] = unirlm-02/.
    """
    return Path(__file__).resolve().parents[4] / "configs"


# ── Response classifier ──────────────────────────────────────────────────


def classify_dashscope_response(rsp: Any) -> str:
    """Map a DashScope SDK response object (or httpx ``Response`` for the
    happyhorse raw-HTTP path) to a ``KeyPool.release`` kind.

    DashScope SDKs return a response object whose ``.status_code`` may be
    429/5xx without raising — so successful-return ≠ ok-from-the-key's-perspective.

    Returns one of: ``"ok"``, ``"auth_invalid"``, ``"daily_quota"``,
    ``"tps_throttle"``, ``"rate_limit"``, ``"overload_503"``,
    ``"network"``, ``"other"`` — same taxonomy as
    :func:`backends._common.key_pool.classify_error`.

    DashScope SDK does not expose ``Retry-After`` headers, so the tuple
    form of ``with_key`` (kind, cooldown_s) isn't used here; callers
    fall back to the kind's policy-table default cooldown.
    """
    status = getattr(rsp, "status_code", None)
    if status is None:
        # No ``.status_code`` attribute → the caller returned a plain value
        # (str / dict / bytes etc.) instead of an SDK response object. Treat
        # as success — if the caller wanted to flag an error, they'd raise.
        return "ok"
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        status_int = 0
    if status_int == HTTPStatus.OK:
        return "ok"

    code = getattr(rsp, "code", "") or ""
    message = getattr(rsp, "message", "") or ""
    kind = classify_error(RuntimeError(
        f"status={status_int} code={code} message={message}"
    ))
    if kind != "other":
        return kind
    # Fall-back: status-code-based bucketing for unmatched messages.
    if status_int in (401, 403):
        return "auth_invalid"
    if status_int == 429:
        return "rate_limit"
    if status_int in (502, 503, 504):
        return "overload_503" if status_int == 503 else "network"
    if status_int >= 500:
        return "network"
    return "other"


# ── Provider registration ─────────────────────────────────────────────────


DASHSCOPE_PROVIDER = Provider(
    name="dashscope",
    api_key_envs=("DASHSCOPE_API_KEY",),
    api_keys_csv_env="DASHSCOPE_API_KEYS",
    # OpenAI-compat gateway URL for chat.completions (qwen text + qwen-VL).
    # The dashscope SDK (VideoSynthesis / ImageSynthesis / MultiModalConversation)
    # uses its own embedded URL; this base_url is consumed only when an
    # ``OpenAICompatibleAgent``-style caller resolves an endpoint via the
    # provider.
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_submit_parallelism=1,
    rate_limits_path=_repo_configs_dir() / "dashscope_rate_limits.json",
    rate_limits_env_override="DASHSCOPE_RATE_LIMITS_CONFIG",
    response_classifier=classify_dashscope_response,
    notes="Alibaba Model Studio — wan*/wanx*/qwen-image async tasks + qwen LLM chat",
)


register_provider(DASHSCOPE_PROVIDER)


__all__ = ["DASHSCOPE_PROVIDER", "classify_dashscope_response"]
