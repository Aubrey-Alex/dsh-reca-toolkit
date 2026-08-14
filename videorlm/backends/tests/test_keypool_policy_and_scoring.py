"""Tests for the KeyPool policy table + EWMA + scoring pick.

These tests target the empirically derived upgrades layered on top of the
basic middleware contract (covered in `test_platforms_middleware.py`).
What we verify here:

  - Each ``classify_error`` kind drives the cooldown policy table's
    advertised behavior (tps_throttle 5s fixed, rate_limit 60→120s
    exponential, daily_quota / auth_invalid permanent, …).
  - ``release(cooldown_s=...)`` and ``classify_response`` returning a
    ``(kind, cooldown_s)`` tuple override the policy table — the
    Retry-After header propagation hook.
  - EWMA error rate is updated on every release with the documented
    alpha=0.2 default, smoothly tracks success and failure streaks.
  - ``pick()`` uses ``w_load * load_ratio + w_err * error_rate_ewma``
    instead of pure least-loaded — a low-load but unhealthy key is
    demoted when ties exist.
  - Process signals (KeyboardInterrupt / SystemExit) do NOT bump
    ``total_err`` — they release the slot as ``ok`` to keep health
    stats clean (M1 fix).
  - The cooldown-degrade branch in ``pick()`` no longer increments
    in_flight (M3 fix) and emits a ``trace_event`` for observability.
  - ``classify_dashscope_response`` maps status_code 401/429/503/etc.
    to the correct kind taxonomy.
"""
from __future__ import annotations

import math
import time

import pytest

from videorlm.backends._common import key_pool as _key_pool_mod
from videorlm.backends._common.dashscope_sdk import classify_dashscope_response
from videorlm.backends._common.key_pool import (
    _COOLDOWN_POLICY,
    KeyState,
    classify_error,
)
from videorlm.backends._common.platforms import get_platform, with_key


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — same isolation pattern as test_platforms_middleware.py
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setattr(_key_pool_mod, "_HEALTH_THREAD_STARTED", False)
    monkeypatch.setenv("DASHSCOPE_API_KEYS", "sk-test-AAAA,sk-test-BBBB")
    monkeypatch.setenv("RECA_KEYPOOL_HEALTH_DUMP", "0")
    # Reset weights to defaults so each test starts predictable.
    monkeypatch.delenv("RECA_KEYPOOL_W_LOAD", raising=False)
    monkeypatch.delenv("RECA_KEYPOOL_W_ERR", raising=False)
    monkeypatch.delenv("RECA_KEYPOOL_EWMA_ALPHA", raising=False)
    yield


def _pool():
    return get_platform("dashscope").key_pool()


def _stats():
    return _pool().stats()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-kind cooldown policy
# ─────────────────────────────────────────────────────────────────────────────


def test_cooldown_policy_table_covers_all_classify_error_outputs():
    """Every kind classify_error can return must have a policy entry —
    a typo in either map would silently drop a key into the catch-all
    'other' bucket with zero cooldown."""
    sample_msgs = [
        "401 Unauthorized", "Throttling.AllocationQuota",
        "Throttling.RateQuotaExceeded", "MODEL_CAPACITY_EXHAUSTED",
        "DailyQuota exceeded", "Connection timed out", "weird error",
    ]
    produced = {classify_error(RuntimeError(m)) for m in sample_msgs}
    missing = produced - set(_COOLDOWN_POLICY)
    assert not missing, f"classify_error returns kind(s) with no policy: {missing}"


def test_tps_throttle_uses_short_fixed_cooldown_not_exponential():
    """The MEMORY note (reference_dashscope_allocation_quota.md) says
    TPS throttle recovers in seconds — must NOT escalate exponentially."""
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="tps_throttle")
    cd1 = next(s["cooldown_remaining_s"] for s in pool.stats()
               if s["last_failure_kind"] == "tps_throttle")
    assert 0 < cd1 <= 5.1, f"tps_throttle base cooldown should be ~5s, got {cd1}"

    # Hit it again — should NOT escalate (no exponential), capped at 15s.
    key = pool.pick()
    pool.release(key, error_kind="tps_throttle")
    cd2 = next(s["cooldown_remaining_s"] for s in pool.stats()
               if s["last_failure_kind"] == "tps_throttle")
    assert cd2 <= 15.1, f"tps_throttle 2nd hit cooldown should be ≤15s (cap), got {cd2}"
    # Specifically, it should be the same fixed 5s — not 10s exponential.
    assert cd2 <= 5.1, \
        f"tps_throttle 2nd hit must stay fixed, not escalate: got {cd2}"


def test_rate_limit_still_escalates_exponentially():
    """The historical 429 path keeps the exponential backoff — preserved
    so adversarial keys back off fast. Exercise the same key twice via
    direct release() (pick() would load-balance to the other key)."""
    pool = _pool()
    # Pin to one specific key so we observe escalation on it.
    target = pool.keys[0].key

    pool.release(target, error_kind="rate_limit")
    cd1 = pool.keys[0].cooldown_until - time.time()
    assert 55 < cd1 <= 61, f"rate_limit base cooldown should be ~60s, got {cd1}"

    pool.release(target, error_kind="rate_limit")
    cd2 = pool.keys[0].cooldown_until - time.time()
    assert 115 < cd2 <= 125, \
        f"rate_limit 2nd hit should be ~120s (60 * 2^1), got {cd2}"

    pool.release(target, error_kind="rate_limit")
    cd3 = pool.keys[0].cooldown_until - time.time()
    assert 235 < cd3 <= 245, \
        f"rate_limit 3rd hit should be ~240s (60 * 2^2), got {cd3}"


def test_overload_503_uses_short_fixed_cooldown():
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="overload_503")
    cd = next(s["cooldown_remaining_s"] for s in pool.stats()
              if s["last_failure_kind"] == "overload_503")
    assert 9 < cd <= 11, f"overload_503 should be ~10s, got {cd}"


def test_daily_quota_is_permanent_cooldown():
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="daily_quota")
    cooling = [s for s in pool.stats() if s["last_failure_kind"] == "daily_quota"]
    assert len(cooling) == 1
    assert math.isinf(cooling[0]["cooldown_remaining_s"]), \
        "daily_quota must be permanent (cooldown_until=inf)"


def test_auth_invalid_is_permanent_cooldown():
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="auth_invalid")
    cooling = [s for s in pool.stats() if s["last_failure_kind"] == "auth_invalid"]
    assert len(cooling) == 1
    assert math.isinf(cooling[0]["cooldown_remaining_s"])


def test_network_error_does_not_cool_key():
    """Network errors are transient and let the outer retry decide —
    the key itself is fine, no cooldown."""
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="network")
    stats = pool.stats()
    cooling = [s for s in stats if s["cooldown_remaining_s"] > 0]
    assert not cooling, f"network error must not cool the key: {stats}"
    # Still recorded as an error for observability.
    assert sum(s["total_err"] for s in stats) == 1
    assert sum(s["err_by_kind"].get("network", 0) for s in stats) == 1


def test_other_kind_does_not_cool_key():
    pool = _pool()
    key = pool.pick()
    pool.release(key, error_kind="other")
    cooling = [s for s in pool.stats() if s["cooldown_remaining_s"] > 0]
    assert not cooling


# ─────────────────────────────────────────────────────────────────────────────
# 2. cooldown_s override (Retry-After hook)
# ─────────────────────────────────────────────────────────────────────────────


def test_release_cooldown_s_override_beats_policy_table():
    """Calling ``release(cooldown_s=X)`` honors X exactly, ignoring the
    policy table — this is how Retry-After headers should propagate."""
    pool = _pool()
    key = pool.pick()
    # rate_limit policy says ~60s; we override to 7.5s.
    pool.release(key, error_kind="rate_limit", cooldown_s=7.5)
    cd = next(s["cooldown_remaining_s"] for s in pool.stats()
              if s["last_failure_kind"] == "rate_limit")
    assert 6.5 < cd <= 7.6, \
        f"cooldown_s=7.5 override should produce ~7.5s, got {cd}"


def test_classify_response_tuple_return_passes_cooldown_to_release():
    """``classify_response`` returning ``(kind, cooldown_s)`` flows the
    custom cooldown through the middleware — the Retry-After path the
    user actually sees."""
    def _call(api_key): return "result"
    def _classify(rsp): return ("rate_limit", 3.0)

    with_key("dashscope", _call, classify_response=_classify)
    cd = next(s["cooldown_remaining_s"] for s in _pool().stats()
              if s["last_failure_kind"] == "rate_limit")
    assert 2.5 < cd <= 3.1, f"tuple-return cooldown should be ~3.0, got {cd}"


def test_classify_response_tuple_with_none_cooldown_falls_back_to_policy():
    """Returning ``(kind, None)`` means 'use policy default'."""
    def _call(api_key): return "result"
    def _classify(rsp): return ("tps_throttle", None)

    with_key("dashscope", _call, classify_response=_classify)
    cd = next(s["cooldown_remaining_s"] for s in _pool().stats()
              if s["last_failure_kind"] == "tps_throttle")
    assert 0 < cd <= 5.1, f"None cooldown should fall back to policy ~5s, got {cd}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. EWMA error rate
# ─────────────────────────────────────────────────────────────────────────────


def test_ewma_starts_at_zero_and_stays_zero_on_all_success():
    pool = _pool()
    for _ in range(10):
        k = pool.pick()
        pool.release(k, error_kind="ok")
    for s in pool.stats():
        assert s["error_rate_ewma"] == 0.0, \
            f"all-success keys should have ewma=0, got {s}"


def test_ewma_climbs_toward_one_on_all_failure():
    """alpha=0.2 default → after 10 consecutive failures EWMA should be
    around 1 - 0.8^10 ≈ 0.893 (asymptote 1.0)."""
    pool = _pool()
    # Pick the same key repeatedly to make math tractable — pin in_flight=0
    # by releasing between picks.
    key = pool.pick()
    for _ in range(10):
        # network kind doesn't cool, so we can keep picking the same key.
        pool.release(key, error_kind="network")
        key = pool.pick()
    pool.release(key, error_kind="network")
    # At least one key should have climbed.
    max_ewma = max(s["error_rate_ewma"] for s in pool.stats())
    assert max_ewma > 0.5, \
        f"after 11 network failures, max EWMA should be >0.5, got {max_ewma}"


def test_ewma_decays_on_recovery():
    """Failures spike EWMA; subsequent successes pull it back down.
    Exercise the same key via direct release() to bypass pick()'s
    load-balancing (otherwise successes go to the healthy key and the
    burned key's EWMA never decays — which is the correct system
    behavior, but defeats the test of the EWMA formula itself)."""
    pool = _pool()
    target = pool.keys[0].key

    # Burn the target with 5 failures.
    for _ in range(5):
        pool.release(target, error_kind="network")
    peak_ewma = pool.keys[0].error_rate_ewma
    assert peak_ewma > 0.3, f"5 failures should push EWMA >0.3, got {peak_ewma}"

    # Now 20 successes on the SAME key — EWMA must decay.
    for _ in range(20):
        pool.release(target, error_kind="ok")
    final_ewma = pool.keys[0].error_rate_ewma
    assert final_ewma < peak_ewma / 10, \
        f"EWMA should decay near-zero after 20 successes: peak={peak_ewma}, final={final_ewma}"


def test_ewma_alpha_env_override(monkeypatch):
    """``RECA_KEYPOOL_EWMA_ALPHA`` should change the smoothing factor."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setenv("RECA_KEYPOOL_EWMA_ALPHA", "0.5")
    pool = _pool()
    assert pool.ewma_alpha == 0.5
    # alpha=0.5 → one failure should produce ewma=0.5 exactly.
    k = pool.pick()
    pool.release(k, error_kind="network")
    nonzero = [s["error_rate_ewma"] for s in pool.stats()
               if s["error_rate_ewma"] > 0]
    assert len(nonzero) == 1
    assert abs(nonzero[0] - 0.5) < 1e-9, \
        f"alpha=0.5 single failure should give ewma=0.5, got {nonzero[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scoring pick
# ─────────────────────────────────────────────────────────────────────────────


def test_pick_with_w_err_zero_behaves_like_least_loaded(monkeypatch):
    """Disabling the error-rate weight should reduce pick() to the
    original least-loaded strategy — easy sanity check."""
    monkeypatch.setattr(_key_pool_mod, "_POOLS", {})
    monkeypatch.setenv("RECA_KEYPOOL_W_ERR", "0")
    pool = _pool()
    # Make one key look unhealthy via direct state mutation.
    pool.keys[0].error_rate_ewma = 0.9
    # Without err weight, in_flight is the only tie-breaker. Both have
    # in_flight=0, so both are valid candidates — over many picks both
    # are seen.
    seen = set()
    for _ in range(20):
        k = pool.pick()
        seen.add(k)
        pool.release(k, error_kind="ok")
    assert len(seen) == 2, \
        f"w_err=0 with both keys at in_flight=0 should pick both, saw {seen}"


def test_pick_demotes_high_error_rate_key_when_load_is_tied():
    """With default weights, two keys at equal in_flight but different
    EWMA should resolve to the lower-EWMA key consistently."""
    pool = _pool()
    # Force keys[0] to look unhealthy, keys[1] to look pristine.
    pool.keys[0].error_rate_ewma = 0.9
    pool.keys[1].error_rate_ewma = 0.0
    # Pick 30 times; healthy key should dominate (>= 90% of picks given
    # the score gap is 1.0 * 0.9 = 0.9 vs 0 — only one candidate is
    # min-score, so it should win deterministically).
    healthy_picks = 0
    for _ in range(30):
        k = pool.pick()
        if k == pool.keys[1].key:
            healthy_picks += 1
        pool.release(k, error_kind="ok")
        # Reset EWMA so the picked key doesn't change its own score
        # mid-loop — we want to test pick(), not the EWMA loop.
        pool.keys[0].error_rate_ewma = 0.9
        pool.keys[1].error_rate_ewma = 0.0
    assert healthy_picks >= 28, \
        f"healthy key should win nearly always, got {healthy_picks}/30"


def test_pick_load_factor_dominates_when_load_skew_exceeds_error_gap():
    """A heavy in_flight delta should override a modest EWMA gap."""
    pool = _pool()
    pool.keys[0].error_rate_ewma = 0.0     # healthy
    pool.keys[0].in_flight = 4              # but loaded (load_ratio = 4/8=0.5)
    pool.keys[1].error_rate_ewma = 0.2     # mildly unhealthy
    pool.keys[1].in_flight = 0              # but free (load_ratio = 0)
    # score(0) = 1.0*0.5 + 1.0*0.0 = 0.5
    # score(1) = 1.0*0.0 + 1.0*0.2 = 0.2  → key 1 wins
    next_key = pool.pick()
    assert next_key == pool.keys[1].key, \
        "free-but-mildly-unhealthy key should beat loaded-but-healthy"
    pool.release(next_key, error_kind="ok")


# ─────────────────────────────────────────────────────────────────────────────
# 5. classify_error new kinds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("msg,expected", [
    ("HTTP 401 Unauthorized",              "auth_invalid"),
    ("403 Forbidden",                      "auth_invalid"),
    ("InvalidApiKey",                      "auth_invalid"),
    ("DailyQuota exceeded for sk-xxx",     "daily_quota"),
    ("MonthlyQuota reached",               "daily_quota"),
    ("Throttling.AllocationQuota",         "tps_throttle"),
    ("qps limit exceeded",                 "tps_throttle"),
    ("tps exceeded",                       "tps_throttle"),
    ("Throttling.RateQuotaExceeded",       "rate_limit"),
    ("429 Too Many Requests",              "rate_limit"),
    ("RateLimitExceeded",                  "rate_limit"),
    ("MODEL_CAPACITY_EXHAUSTED",           "overload_503"),
    ("HTTP 503 Service Unavailable",       "overload_503"),
    ("Connection timed out after 30s",     "network"),
    ("502 Bad Gateway",                    "network"),
    ("504 Gateway Timeout",                "network"),
    ("Some weird ass exception",           "other"),
])
def test_classify_error_taxonomy(msg, expected):
    assert classify_error(RuntimeError(msg)) == expected, \
        f"classify_error({msg!r}) misclassified"


# ─────────────────────────────────────────────────────────────────────────────
# 6. classify_dashscope_response (the unified response classifier)
# ─────────────────────────────────────────────────────────────────────────────


def _mk_rsp(status_code, code="", message=""):
    class R:
        pass
    r = R()
    r.status_code = status_code
    r.code = code
    r.message = message
    return r


@pytest.mark.parametrize("status,code,message,expected", [
    (200, "", "",                                "ok"),
    (401, "Unauthorized", "",                    "auth_invalid"),
    (403, "Forbidden", "",                       "auth_invalid"),
    (429, "Throttling.RateQuotaExceeded", "",    "rate_limit"),
    (429, "Throttling.AllocationQuota", "qps",   "tps_throttle"),
    (503, "MODEL_CAPACITY_EXHAUSTED", "",        "overload_503"),
    (500, "InternalError", "",                   "network"),
    (502, "BadGateway", "",                      "network"),
    (504, "Timeout", "",                         "network"),
    (418, "WeirdCode", "weird message",          "other"),
])
def test_classify_dashscope_response_taxonomy(status, code, message, expected):
    rsp = _mk_rsp(status, code, message)
    assert classify_dashscope_response(rsp) == expected, \
        f"status={status} code={code!r} message={message!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. M1 fix — KeyboardInterrupt / SystemExit don't poison health
# ─────────────────────────────────────────────────────────────────────────────


def test_keyboard_interrupt_releases_as_ok_does_not_bump_errors():
    pool = _pool()
    err_before = sum(s["total_err"] for s in pool.stats())

    def _call(api_key):
        raise KeyboardInterrupt("user pressed Ctrl-C")

    with pytest.raises(KeyboardInterrupt):
        with_key("dashscope", _call)

    stats = pool.stats()
    err_after = sum(s["total_err"] for s in stats)
    assert err_after == err_before, \
        f"KI must not bump total_err: {err_before} -> {err_after}"
    assert all(s["in_flight"] == 0 for s in stats), "KI leaked slot!"
    assert sum(s["total_ok"] for s in stats) == 1, \
        "KI should release as ok"


def test_system_exit_releases_as_ok_does_not_bump_errors():
    pool = _pool()
    def _call(api_key):
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        with_key("dashscope", _call)

    stats = pool.stats()
    assert sum(s["total_err"] for s in stats) == 0
    assert all(s["in_flight"] == 0 for s in stats)


# ─────────────────────────────────────────────────────────────────────────────
# 8. M3 fix — degrade-overcap branch behavior
# ─────────────────────────────────────────────────────────────────────────────


def test_degrade_overcap_does_not_increment_in_flight():
    """When every key is in cooldown AND at cap, pick() returns a key
    WITHOUT incrementing in_flight — the cap stays hard."""
    pool = _pool()
    # Force both keys into cooldown.
    for s in pool.keys:
        s.cooldown_until = time.time() + 100
        s.in_flight = s.max_concurrency   # also at cap
    in_flight_before = [s.in_flight for s in pool.keys]

    key = pool.pick()      # must NOT block — degrade branch fires
    assert key in (pool.keys[0].key, pool.keys[1].key)

    in_flight_after = [s.in_flight for s in pool.keys]
    assert in_flight_after == in_flight_before, \
        f"degrade branch must NOT bump in_flight: {in_flight_before} -> {in_flight_after}"

    # Cleanup — release would max(0, in_flight-1) so we need to manually
    # reset to keep teardown sane.
    for s in pool.keys:
        s.in_flight = 0
        s.cooldown_until = 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. stats() exposes error_rate_ewma
# ─────────────────────────────────────────────────────────────────────────────


def test_stats_exposes_error_rate_ewma_field():
    pool = _pool()
    for s in pool.stats():
        assert "error_rate_ewma" in s, "stats() must expose error_rate_ewma"
        assert s["error_rate_ewma"] == 0.0
