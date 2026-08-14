"""Per-model concurrency cap + min-interval pacing — provider-agnostic.

Sits between ``with_key``'s KeyPool layer and the actual network submit:

  - **KeyPool**: picks a healthy key (cross-model, per-provider).
  - **RateLimiter**: enforces per-model token-bucket — capped concurrent
    in-flight submits + min interval between submits to the same model.
  - **Network call**: actually issues the HTTP request with the picked key.

Each provider gets ONE cached RateLimiter (via ``cached_rate_limiter``,
mirroring the ``cached_key_pool`` pattern). The model→limits table comes
from the provider's ``rate_limits()`` (loaded from the bundled JSON,
overlayable by env). Unknown models fall back to the limiter's defaults.

Previously this lived in ``dashscope_sdk.py`` and only knew dashscope
models — now any provider with a rate-limits JSON gets per-model gating
automatically (e.g. ``with_key("openai", call, model="gpt-image-2")``
honors the 2.0 s Cloudflare-safe interval from
``configs/openai_rate_limits.json``).
"""
from __future__ import annotations

import threading
import time


# Defaults when a model isn't in the provider's rate_limits() table.
# Hardcoded — provider's JSON is the configuration surface; env vars
# are reserved for provider selection + API keys + base URLs only.
DEFAULT_INTERVAL_S = 0.3
DEFAULT_MAX_PARALLEL = 8


class _RateKeyState:
    """Per-model concurrency semaphore + last-submit timestamp."""

    __slots__ = ("interval", "max_parallel", "lock", "sem", "last_call")

    def __init__(self, interval: float, max_parallel: int) -> None:
        self.interval = interval
        self.max_parallel = max(1, int(max_parallel))
        self.lock = threading.Lock()
        self.sem = threading.BoundedSemaphore(self.max_parallel)
        self.last_call = 0.0


class RateLimiter:
    """Per-provider rate limiter. ``acquire(model)`` blocks until a slot
    is free AND the min-interval has elapsed since the last submit of
    the same model. Always paired with a ``release(state)`` in
    ``finally`` (or use the contextmanager API on top via ``with_key``).
    """

    def __init__(
        self,
        provider_name: str = "",
        *,
        default_interval: float = DEFAULT_INTERVAL_S,
        default_parallel: int = DEFAULT_MAX_PARALLEL,
    ) -> None:
        self.provider_name = provider_name
        self.default_interval = float(default_interval)
        self.default_parallel = max(1, int(default_parallel))
        self._states: dict[str, _RateKeyState] = {}
        self._states_lock = threading.Lock()

    # ── table resolution (live — picks up provider's reloaded JSON) ──────

    def _table(self) -> dict[str, dict[str, float | int]]:
        if not self.provider_name:
            return {}
        try:
            from .providers import get_provider
            return get_provider(self.provider_name).rate_limits()
        except Exception:  # noqa: BLE001
            return {}

    def _resolve_limits(self, model: str) -> tuple[float, int]:
        cfg = self._table().get(model) or {}
        interval = float(cfg.get("interval", self.default_interval))
        max_parallel = int(cfg.get("max_parallel", self.default_parallel))
        return interval, max(1, max_parallel)

    def _get_state(self, model: str) -> _RateKeyState:
        interval, max_parallel = self._resolve_limits(model)
        with self._states_lock:
            state = self._states.get(model)
            # Reload the state if either field changed (e.g. JSON was hot-swapped).
            if (
                state is None
                or state.interval != interval
                or state.max_parallel != max_parallel
            ):
                state = _RateKeyState(interval, max_parallel)
                self._states[model] = state
            return state

    # ── acquire / release ────────────────────────────────────────────────

    def acquire(self, model: str) -> _RateKeyState:
        state = self._get_state(model)
        state.sem.acquire()
        with state.lock:
            now = time.time()
            gap = now - state.last_call
            if gap < state.interval:
                time.sleep(state.interval - gap)
            state.last_call = time.time()
        return state

    @staticmethod
    def release(state: _RateKeyState) -> None:
        state.sem.release()


# ── cached factory (one limiter per provider, process-wide) ──────────────


_LIMITERS: dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def cached_rate_limiter(provider_name: str) -> RateLimiter:
    """Return the singleton ``RateLimiter`` for ``provider_name``.

    Same instance for every caller in the process so per-model state
    (semaphore + last_call) is shared across the codebase.
    """
    key = (provider_name or "").strip().lower()
    with _LIMITERS_LOCK:
        rl = _LIMITERS.get(key)
        if rl is None:
            rl = RateLimiter(provider_name=key)
            _LIMITERS[key] = rl
        return rl


def reset_rate_limiters_for_tests() -> None:
    """Clear the per-process limiter cache. ONLY for test isolation."""
    with _LIMITERS_LOCK:
        _LIMITERS.clear()


__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_PARALLEL",
    "RateLimiter",
    "cached_rate_limiter",
    "reset_rate_limiters_for_tests",
]
