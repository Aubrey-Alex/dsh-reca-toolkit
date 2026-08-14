"""OpenAI(-compatible) provider — covers openai.com, any self-hosted
OpenAI-compatible gateway, and anything that speaks the OpenAI Chat /
Images API contract.

Routing is env-driven: the consumer sets ``OPENAI_BASE_URL`` to point
at whichever gateway they want (omit to default to openai.com).

Multi-key: list keys via either
  - ``OPENAI_API_KEYS=k1,k2,...`` (CSV; preferred for multi-key setups), OR
  - ``OPENAI_API_KEY=k1`` (single-key form).
  KeyPool merges everything it finds into one rotating pool.

Rate limits source: ``configs/openai_rate_limits.json``. Override the
table per-deployment via ``OPENAI_RATE_LIMITS_CONFIG=<path>``.
"""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from ..key_pool import classify_error
from . import register_provider
from ._base import Provider


def _repo_configs_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "configs"


# ── Response classifier ──────────────────────────────────────────────────


def classify_openai_response(rsp: Any) -> str:
    """Map an OpenAI-compatible response object (or ``httpx.Response``
    for raw HTTP paths) to a ``KeyPool.release`` kind.

    Duck-typed on:
      - ``.status_code`` (int) — primary signal
      - ``.headers["x-ratelimit-*"]`` or ``.json().get("error")`` — best-effort

    OpenAI's ``Retry-After`` header IS exposed (unlike DashScope), but we
    don't parse it here — callers wanting precise cooldown should wrap
    the return in a ``classify_response`` tuple form themselves.
    """
    status = getattr(rsp, "status_code", None)
    if status is None:
        # Caller returned a plain value (str / dict / SDK object that
        # already raised on error) → treat as success.
        return "ok"
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        status_int = 0
    if status_int == HTTPStatus.OK:
        return "ok"

    # OpenAI returns 401 for invalid key, 429 for rate limit / quota
    # exceeded, 5xx for transient server / Cloudflare issues.
    if status_int in (401, 403):
        return "auth_invalid"
    if status_int == 429:
        # Could be rate_limit OR quota. Body has the discriminator.
        # Cheap heuristic: peek at body text.
        body = ""
        try:
            body = (getattr(rsp, "text", "") or "").lower()
        except Exception:  # noqa: BLE001
            pass
        if "quota" in body or "billing" in body:
            return "daily_quota"
        return "rate_limit"
    if status_int == 503:
        return "overload_503"
    if status_int in (502, 504, 524):
        # 524 is Cloudflare-specific; group with network for short cool-off.
        return "network"
    if status_int >= 500:
        return "network"

    # Fallback: synthesize an error from any available text for the
    # central classifier to bucket.
    text = ""
    try:
        text = getattr(rsp, "text", "") or ""
    except Exception:  # noqa: BLE001
        pass
    return classify_error(RuntimeError(f"status={status_int} body={text[:200]}"))


# ── Provider registration ─────────────────────────────────────────────────


OPENAI_PROVIDER = Provider(
    name="openai",
    api_key_envs=("OPENAI_API_KEY",),
    api_keys_csv_env="OPENAI_API_KEYS",
    # Leave empty; consumers read OPENAI_BASE_URL at call time (env is the
    # source of truth for routing).
    base_url="",
    default_submit_parallelism=8,
    rate_limits_path=_repo_configs_dir() / "openai_rate_limits.json",
    rate_limits_env_override="OPENAI_RATE_LIMITS_CONFIG",
    response_classifier=classify_openai_response,
    notes=(
        "Generic OpenAI-compatible endpoint. Route via OPENAI_BASE_URL env "
        "(any gateway; omit for openai.com). Multi-key via OPENAI_API_KEYS "
        "CSV or single OPENAI_API_KEY."
    ),
)


register_provider(OPENAI_PROVIDER)


__all__ = ["OPENAI_PROVIDER", "classify_openai_response"]
