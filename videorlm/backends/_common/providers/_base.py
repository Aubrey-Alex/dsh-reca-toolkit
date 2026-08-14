"""Provider abstraction — bundles per-provider metadata, key pool, rate
limits, and response classifier in one place.

Replaces the simpler ``PlatformProfile`` (which had only env names +
base_url). The richer ``Provider`` lets ``with_key`` middleware auto-pull
the right ``classify_response`` callback without each call site re-passing
it, and lets ``RateLimiter`` consume a per-provider JSON table instead of
hardcoding a dashscope-only ``_MODEL_LIMITS``.

Lookup is keyed by short provider name (``"dashscope"``, ``"gateway"``,
``"openai"``, ``"kling"``, ``"pixverse"``, ``"vidu"``).

Each provider concrete module registers itself in ``providers.__init__``;
see ``dashscope.py`` / ``openai.py`` for examples.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..env import env_value
from ..key_pool import KeyPool, cached_key_pool


@dataclass(frozen=True)
class Provider:
    """Bundles everything ``with_key`` and ``RateLimiter`` need.

    Compared to the legacy ``PlatformProfile``, this adds:
      - ``rate_limits_path``: optional JSON file with per-model
        ``{interval_s, max_parallel}`` defaults. Loaded lazily, cached on
        first ``rate_limits()`` call.
      - ``response_classifier``: callable mapping SDK / HTTP response →
        ``KeyPool.release`` kind string. ``with_key`` auto-uses this when
        the caller passes ``classify_response=None``.
    """

    name: str
    api_key_envs: tuple[str, ...]
    api_keys_csv_env: str = ""
    base_url: str = ""
    default_submit_parallelism: int = 1
    rate_limits_path: Path | None = None
    # Env var name pointing at a custom JSON to swap in for
    # ``rate_limits_path``. When the env is set + the file exists, that
    # file is loaded instead of the bundled path. Example:
    # ``rate_limits_env_override="DASHSCOPE_RATE_LIMITS_CONFIG"``.
    # NOTE: this is a *file-path* selector, not a per-value override —
    # actual rate-limit values live in JSON only (no env knobs for
    # interval / max_parallel). Edit the JSON to change limits.
    rate_limits_env_override: str = ""
    response_classifier: Callable[[Any], str] | None = None
    notes: str = ""

    # Back-compat alias for the legacy ``PlatformProfile.provider`` field;
    # 11+ call sites in ``media/impl/`` use ``_PLATFORM.provider`` to set
    # ``BackendCapabilities.provider=...``. Keep the attribute spelling.
    @property
    def provider(self) -> str:
        return self.name

    # ── auth + key pool ──────────────────────────────────────────────────

    def api_key(self) -> str:
        """Resolve a single api_key for bootstrap (e.g. OpenAI client
        construction). Lookup order:

          1. Each name in ``api_key_envs`` (singular envs, in order).
          2. The first key from the CSV at ``api_keys_csv_env`` (so
             multi-key setups don't need to also set a singular env
             just to satisfy a caller that wants one).
        """
        for env in self.api_key_envs:
            val = env_value(env)
            if val:
                return val
        if self.api_keys_csv_env:
            csv = env_value(self.api_keys_csv_env).strip()
            if csv:
                first = csv.split(",")[0].strip()
                if first:
                    return first
        return ""

    def key_pool(self) -> KeyPool:
        """Return the cached ``KeyPool`` for this provider. Same instance
        across all callers in the process (provider name is the cache key)."""
        return cached_key_pool(
            self.name,
            self.api_key_envs,
            default_csv_env=self.api_keys_csv_env or None,
        )

    # ── rate limits (per-model JSON, lazy + cached) ──────────────────────

    def rate_limits(self) -> dict[str, dict[str, float | int]]:
        """Load ``{model_name: {"interval": float, "max_parallel": int}}``.

        Resolution:

          1. If ``rate_limits_env_override`` env is set + the file
             exists + loads to a non-empty dict → use that custom JSON.
          2. Otherwise (env unset, path missing, or malformed/empty) →
             fall back to ``rate_limits_path`` (bundled JSON).
          3. If neither is available → ``{}``.

        Rate-limit values come from JSON only — there is no env
        override at the value level. To customize, edit the JSON or
        point ``rate_limits_env_override`` at a different file."""
        if self.rate_limits_env_override:
            raw = os.environ.get(self.rate_limits_env_override, "").strip()
            if raw:
                env_path = Path(raw).expanduser().resolve()
                if env_path.exists():
                    custom = _load_rate_limits(env_path)
                    if custom:
                        return custom
        return _load_rate_limits(self.rate_limits_path)

    # ── response classifier (default: always "ok") ───────────────────────

    def classify_response(self, rsp: Any) -> str:
        if self.response_classifier is None:
            return "ok"
        return self.response_classifier(rsp)


# ── JSON loader (provider-agnostic helper) ───────────────────────────────


def _load_rate_limits(path: Path | None) -> dict[str, dict[str, float | int]]:
    """Read a rate-limits JSON file ``{"models": {<name>: {interval_s, max_parallel}, ...}}``.

    Silent fall-throughs on every error path so a typo in one config
    can't take down the whole provider system.
    """
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    models = data.get("models") or {}
    if not isinstance(models, dict):
        return {}
    out: dict[str, dict[str, float | int]] = {}
    for name, cfg in models.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        interval = cfg.get("interval_s")
        max_parallel = cfg.get("max_parallel")
        if interval is None or max_parallel is None:
            continue
        try:
            out[name] = {
                "interval": float(interval),
                "max_parallel": max(1, int(max_parallel)),
            }
        except (TypeError, ValueError):
            continue
    return out


__all__ = ["Provider"]
