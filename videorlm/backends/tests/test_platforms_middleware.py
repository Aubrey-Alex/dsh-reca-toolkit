"""Unit tests for ``backends._common.platforms`` — the shared
provider-metadata + ``with_key`` middleware.

The middleware contract is small but load-bearing: every DashScope call
across ``media/`` and ``llm/`` now funnels through ``with_key`` and
relies on it to (a) pick a healthy key, (b) call the wrapped function
exactly once, (c) classify the outcome, (d) release the key with the
right ``error_kind`` so cooldown/health tracking stays correct.

These tests run pure in-process — no DashScope network — by injecting
fake keys via ``DASHSCOPE_API_KEYS`` env and asserting against
``KeyPool.stats()`` after each call.
"""
from __future__ import annotations

import threading
import time

import pytest

# Reach into the private pool cache to isolate tests from any earlier
# import that already warmed up the dashscope pool.
from videorlm.backends._common import key_pool as _key_pool_mod
from videorlm.backends._common.platforms import (
    PlatformProfile,
    get_platform,
    with_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_pools_and_env(monkeypatch):
    """Each test starts with an empty process-wide pool cache and known env.

    Without this, the first test to call ``get_platform("dashscope")``
    would lock the pool to whatever env state happened to be live at
    import time and every later test would inherit that pool's stats.
    """
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "sk-test-aaaa,sk-test-bbbb")
    # Tight cooldown so we don't accidentally wait between tests if a
    # rate_limit classification fires.
    monkeypatch.setenv("RECA_KEYPOOL_COOLDOWN_S", "1")
    # Disable the health dumper daemon — it would otherwise leak across
    # tests and print to stdout.
    monkeypatch.setenv("RECA_KEYPOOL_HEALTH_DUMP", "0")
    yield


# ─────────────────────────────────────────────────────────────────────────────
# Static metadata
# ─────────────────────────────────────────────────────────────────────────────


def test_get_platform_dashscope_returns_profile():
    profile = get_platform("dashscope")
    assert isinstance(profile, PlatformProfile)
    assert profile.provider == "dashscope"


def test_dashscope_api_key_env_is_canonical():
    """After the 2026-05 env cleanup, the dashscope profile resolves keys
    from ``DASHSCOPE_API_KEY`` only; multi-key fan-out lives in
    ``DASHSCOPE_API_KEYS``. Legacy aliases (``QWEN_API_KEY`` /
    ``dashscope_new_api_key``) are no longer probed."""
    profile = get_platform("dashscope")
    assert profile.api_key_envs == ("DASHSCOPE_API_KEY",)
    assert profile.api_keys_csv_env == "DASHSCOPE_API_KEYS"


def test_get_platform_unknown_provider_raises():
    with pytest.raises(KeyError, match="unknown provider"):
        get_platform("not-a-real-provider")


def test_get_platform_case_insensitive():
    assert get_platform("DashScope").provider == "dashscope"
    assert get_platform("  openai  ").provider == "openai"


# ─────────────────────────────────────────────────────────────────────────────
# Pool caching / sharing — the central claim of the refactor
# ─────────────────────────────────────────────────────────────────────────────


def test_key_pool_is_cached_per_provider():
    p1 = get_platform("dashscope").key_pool()
    p2 = get_platform("dashscope").key_pool()
    assert p1 is p2, "platform.key_pool() must return the cached singleton"


def test_media_and_llm_share_same_dashscope_pool():
    """media/impl/dashscope/_platform.dashscope_key_pool() and any
    with_key('dashscope', ...) caller must hit the SAME KeyPool — that's
    the whole point of lifting platforms.py to _common/."""
    from videorlm.backends.media.impl.dashscope._platform import (
        dashscope_key_pool as media_pool_accessor,
    )

    media_pool = media_pool_accessor()

    captured: dict = {}

    def _capture(api_key: str) -> str:
        captured["pool_id_during_call"] = id(get_platform("dashscope").key_pool())
        return "ok"

    with_key("dashscope", _capture)
    assert captured["pool_id_during_call"] == id(media_pool)


# ─────────────────────────────────────────────────────────────────────────────
# with_key middleware — happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_with_key_invokes_call_with_a_real_key():
    seen_keys: list[str] = []

    def _call(api_key: str) -> str:
        seen_keys.append(api_key)
        return "ok"

    result = with_key("dashscope", _call)
    assert result == "ok"
    assert len(seen_keys) == 1
    assert seen_keys[0] in ("sk-test-aaaa", "sk-test-bbbb")


def test_with_key_releases_key_after_success():
    def _call(api_key: str) -> str:
        return "fine"

    with_key("dashscope", _call)

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["in_flight"] == 0 for s in stats), \
        f"key not released: {stats}"
    assert sum(s["total_ok"] for s in stats) == 1
    assert sum(s["total_err"] for s in stats) == 0


def test_with_key_load_balances_across_keys():
    """With 2 keys, sequential calls should exercise both because pool.pick()
    breaks ties (equal score) randomly. 10 picks gives (1/2)^9 ≈ 0.2 % flake
    rate vs 12.5 % at 4 picks."""
    def _call(api_key: str) -> str:
        return api_key

    seen = [with_key("dashscope", _call) for _ in range(10)]
    unique = set(seen)
    assert unique <= {"sk-test-aaaa", "sk-test-bbbb"}
    assert len(unique) == 2, f"only saw {unique}, expected both keys"


# ─────────────────────────────────────────────────────────────────────────────
# with_key middleware — exception path
# ─────────────────────────────────────────────────────────────────────────────


def test_with_key_releases_key_on_exception_and_reraises():
    class Boom(RuntimeError):
        pass

    def _call(api_key: str):
        raise Boom("kaboom")

    with pytest.raises(Boom, match="kaboom"):
        with_key("dashscope", _call)

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["in_flight"] == 0 for s in stats), \
        f"key leaked on exception: {stats}"
    assert sum(s["total_err"] for s in stats) == 1


def test_with_key_rate_limit_exception_triggers_cooldown():
    """``classify_error`` maps 'Throttling' / 'RateQuota' substrings to
    ``rate_limit``, which sets ``cooldown_until`` on the picked key."""
    def _call(api_key: str):
        raise RuntimeError("Throttling.RateQuotaExceeded")

    with pytest.raises(RuntimeError):
        with_key("dashscope", _call)

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    cooling = [s for s in stats if s["cooldown_remaining_s"] > 0]
    assert len(cooling) == 1, \
        f"expected exactly one key in cooldown, got {stats}"
    assert cooling[0]["last_failure_kind"] == "rate_limit"


# ─────────────────────────────────────────────────────────────────────────────
# with_key middleware — classify_response callback
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_response_rate_limit_cools_key_even_without_raising():
    """DashScope SDKs return response objects with status_code=429 instead
    of raising. The middleware must still cool down the key based on
    ``classify_response``'s verdict — that's the whole reason the hook
    exists."""
    class FakeRsp:
        status_code = 429
        code = "Throttling"
        message = "slow down"

    def _call(api_key: str) -> FakeRsp:
        return FakeRsp()

    def _classify(rsp) -> str:
        return "rate_limit" if rsp.status_code == 429 else "ok"

    result = with_key("dashscope", _call, classify_response=_classify)
    assert result.status_code == 429   # call did NOT raise

    pool = get_platform("dashscope").key_pool()
    cooling = [s for s in pool.stats() if s["cooldown_remaining_s"] > 0]
    assert len(cooling) == 1
    assert cooling[0]["err_by_kind"].get("rate_limit", 0) == 1


def test_classify_response_ok_keeps_key_warm():
    class FakeRsp:
        status_code = 200

    def _call(api_key: str) -> FakeRsp:
        return FakeRsp()

    def _classify(rsp) -> str:
        return "ok"

    with_key("dashscope", _call, classify_response=_classify)

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["cooldown_remaining_s"] == 0 for s in stats)
    assert sum(s["total_ok"] for s in stats) == 1


def test_classify_response_buggy_callback_does_not_mask_successful_call():
    """If classify_response itself raises, the caller's result must still
    be returned and the key released (defaulting to ``ok``). We never
    want a logging bug to swallow a successful API response."""
    def _call(api_key: str) -> str:
        return "real result"

    def _bad_classify(rsp) -> str:
        raise RuntimeError("classifier broke")

    result = with_key("dashscope", _call, classify_response=_bad_classify)
    assert result == "real result"

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["in_flight"] == 0 for s in stats)
    assert sum(s["total_ok"] for s in stats) == 1  # fell back to ok


# ─────────────────────────────────────────────────────────────────────────────
# with_key middleware — concurrency
# ─────────────────────────────────────────────────────────────────────────────


def test_with_key_concurrent_calls_release_all_slots():
    """Stress: 20 threads × 2 keys (per-key cap 8 by default). After all
    threads finish, every key must be back to in_flight=0 — no leak."""
    in_flight_peak: list[int] = [0]
    lock = threading.Lock()

    def _call(api_key: str) -> str:
        pool = get_platform("dashscope").key_pool()
        with lock:
            cur = sum(s["in_flight"] for s in pool.stats())
            if cur > in_flight_peak[0]:
                in_flight_peak[0] = cur
        time.sleep(0.01)
        return "ok"

    threads = [
        threading.Thread(target=lambda: with_key("dashscope", _call))
        for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["in_flight"] == 0 for s in stats), \
        f"slots leaked after concurrent run: {stats}"
    assert sum(s["total_ok"] for s in stats) == 20
    # Sanity: middleware DID actually concurrently hold multiple slots.
    assert in_flight_peak[0] >= 2, \
        f"middleware never held >1 slot (in_flight peak={in_flight_peak[0]})"


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat: dashscope submit_with_key thin shim
# ─────────────────────────────────────────────────────────────────────────────


def test_dashscope_submit_with_key_is_a_thin_shim_over_with_key():
    """``submit_with_key(call)`` should behave identically to
    ``with_key("dashscope", call, classify_response=response_error_kind)``
    — same key release, same classification."""
    from videorlm.backends.media.impl.dashscope._platform import submit_with_key

    class FakeRsp:
        status_code = 200

    def _call(api_key: str) -> FakeRsp:
        return FakeRsp()

    rsp = submit_with_key(_call)
    assert rsp.status_code == 200

    pool = get_platform("dashscope").key_pool()
    stats = pool.stats()
    assert all(s["in_flight"] == 0 for s in stats)
    assert sum(s["total_ok"] for s in stats) == 1
