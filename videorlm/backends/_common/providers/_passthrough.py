"""Third-party media providers exposed through the DashScope gateway
(kling / pixverse / vidu).

They share the same API key envs as ``dashscope`` (the request routes
through the same gateway, just with a different ``model`` field), but
keep a distinct provider identity so:
  - ``BackendCapabilities.provider`` reflects the upstream vendor for
    UI / cost-attribution / observability.
  - Per-vendor rate-limit JSON can be added later without touching the
    dashscope table.

No rate_limits_path / response_classifier wired up yet — these
providers haven't been exercised in production paths. ``with_key``
falls back to the no-op default classifier; ``RateLimiter`` falls back
to the global default interval / max_parallel.
"""
from __future__ import annotations

from . import register_provider
from ._base import Provider


_SHARED_DASHSCOPE_KEY_ENVS = ("DASHSCOPE_API_KEY",)
_SHARED_CSV_ENV = "DASHSCOPE_API_KEYS"


KLING_PROVIDER = Provider(
    name="kling",
    api_key_envs=_SHARED_DASHSCOPE_KEY_ENVS,
    api_keys_csv_env=_SHARED_CSV_ENV,
    default_submit_parallelism=4,
    notes="Kling models exposed through the DashScope gateway",
)

PIXVERSE_PROVIDER = Provider(
    name="pixverse",
    api_key_envs=_SHARED_DASHSCOPE_KEY_ENVS,
    api_keys_csv_env=_SHARED_CSV_ENV,
    default_submit_parallelism=4,
    notes="PixVerse models exposed through the DashScope gateway",
)

VIDU_PROVIDER = Provider(
    name="vidu",
    api_key_envs=_SHARED_DASHSCOPE_KEY_ENVS,
    api_keys_csv_env=_SHARED_CSV_ENV,
    default_submit_parallelism=4,
    notes="Vidu models exposed through the DashScope gateway",
)


for _p in (KLING_PROVIDER, PIXVERSE_PROVIDER, VIDU_PROVIDER):
    register_provider(_p)


__all__ = ["KLING_PROVIDER", "PIXVERSE_PROVIDER", "VIDU_PROVIDER"]
