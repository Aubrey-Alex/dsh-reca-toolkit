"""Compat surface for the old ``platforms.py`` API + the ``with_key`` middleware.

The provider-metadata concept moved into ``backends._common.providers``
(see ``providers/_base.py::Provider`` + ``providers/__init__.py`` registry).

This module keeps three symbols stable for callers:

  - ``PlatformProfile``  — alias of ``Provider`` (was the dataclass).
  - ``get_platform``     — alias of ``providers.get_provider``.
  - ``with_key``         — the pick-call-classify-release middleware,
                            now auto-pulls ``classify_response`` from the
                            target provider when the caller omits it.

New code is encouraged to import directly from
``backends._common.providers``; ``platforms.py`` remains the public
surface for the existing 30+ call sites.
"""
from __future__ import annotations

from typing import Callable, TypeVar

from .key_pool import classify_error
from .providers import (
    Provider,
    get_provider,
    list_providers,
    register_provider,
)
from .rate_limiter import cached_rate_limiter

T = TypeVar("T")


# ── Back-compat aliases (DO NOT REMOVE without sweeping every importer) ──


PlatformProfile = Provider
get_platform = get_provider


# ── ``with_key`` middleware ──────────────────────────────────────────────


_ClassifyReturn = str | tuple[str, float | None]


def with_key(
    provider: str,
    call: Callable[[str], T],
    *,
    model: str | None = None,
    classify_response: Callable[[T], _ClassifyReturn] | None = None,
) -> T:
    """Run ``call(api_key)`` inside the provider's KeyPool envelope —
    optionally gated by a per-model RateLimiter.

    Middleware contract:

      1. **Rate gate** (if ``model`` is given): acquire the provider's
         per-model RateLimiter slot. Blocks until the per-model
         concurrency cap is below ``max_parallel`` AND the min-interval
         since the last submit of the same model has elapsed. Limits
         come from ``get_provider(provider).rate_limits()[model]``
         (default fallback used for unknown models).
      2. ``KeyPool.pick()`` blocks until a healthy, under-cap key is available.
      3. ``call(api_key)`` is invoked exactly once with that key.
      4. If ``classify_response`` is None, the **provider's own
         ``classify_response``** (e.g. ``classify_dashscope_response`` for
         dashscope, ``classify_openai_response`` for openai) is used.
         Pass an explicit callable to override per-call (e.g. when you
         need ``Retry-After`` tuple semantics).
      5. The classifier maps the result to a kind string OR
         ``(kind, cooldown_s)`` tuple. Tuple form lets you propagate a
         server-suggested ``Retry-After``.
      6. On exception, :func:`classify_error` maps the exception to a kind;
         the exception is re-raised after release. ``KeyboardInterrupt`` /
         ``SystemExit`` / ``GeneratorExit`` are NOT classified — they
         release the slot with ``ok`` so a Ctrl-C doesn't poison key health.
      7. The key + rate gate are always released in ``finally`` — no
         leaks even if ``classify_response`` itself raises (defaults to
         ``ok`` in that case to avoid swallowing a successful call result).

    This is the single source of truth for "rate-gate → pick → call →
    release". DashScope ``submit_with_retry`` keeps its dashscope-specific
    wrapper but consumes the same ``cached_rate_limiter("dashscope")``;
    OpenAI / future providers get per-model rate limiting just
    by passing ``model="..."``.
    """
    prov = get_provider(provider)
    pool = prov.key_pool()
    classifier = classify_response if classify_response is not None \
        else prov.classify_response

    rl_gate = None
    if model:
        rl_gate = cached_rate_limiter(provider).acquire(model)

    key = pool.pick()
    error_kind = "ok"
    cooldown_s: float | None = None
    try:
        result = call(key)
        try:
            verdict = classifier(result)
        except Exception:  # noqa: BLE001
            # classifier bugs should not mask a successful call.
            verdict = "ok"
        if isinstance(verdict, tuple):
            kind, cd = verdict
            error_kind = kind or "ok"
            cooldown_s = cd
        else:
            error_kind = verdict or "ok"
        return result
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        # Process signals are not a key fault — release as ok so health
        # stats stay clean on Ctrl-C / interpreter shutdown.
        error_kind = "ok"
        raise
    except BaseException as exc:  # noqa: BLE001
        error_kind = classify_error(exc)
        raise
    finally:
        pool.release(key, error_kind=error_kind, cooldown_s=cooldown_s)
        if rl_gate is not None:
            cached_rate_limiter(provider).release(rl_gate)


__all__ = [
    "PlatformProfile",
    "Provider",
    "get_platform",
    "get_provider",
    "list_providers",
    "register_provider",
    "with_key",
]
