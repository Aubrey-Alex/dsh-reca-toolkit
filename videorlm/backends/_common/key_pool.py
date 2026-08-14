"""Thread-safe API-key pool for backend providers.

The pool is intentionally provider-agnostic: callers pick a key before a
network submit, then release it with a fine-grained error kind after the
request returns or raises. The middleware in ``platforms.with_key``
funnels every dashscope/openai-compat call through this contract.

Design:
  - **Pick** is a 2-factor score (``w_load * load_ratio + w_err *
    error_rate_ewma``) over healthy candidates, ties random. A degraded
    key is naturally demoted before its cooldown fires.
  - **Release** is policy-table driven (``_COOLDOWN_POLICY``): each
    error kind has its own ``(base, cap, escalate)`` cooldown — true
    429 escalates exponentially, ``tps_throttle`` is short fixed,
    ``daily_quota`` / ``auth_invalid`` are permanent, ``network`` /
    ``other`` get no cooldown. Callers can override per-call via
    ``release(cooldown_s=<float>)`` (use for ``Retry-After`` headers).
  - **EWMA** tracks sliding error rate per key (alpha env-tunable).

Env knobs:
  - ``RECA_KEYPOOL_PER_KEY_CAP_<PROVIDER>`` / ``RECA_KEYPOOL_PER_KEY_CAP``
    (default 8): per-key concurrency cap.
  - ``RECA_KEYPOOL_W_LOAD`` (default 1.0) / ``RECA_KEYPOOL_W_ERR``
    (default 1.0): pick score weights.
  - ``RECA_KEYPOOL_EWMA_ALPHA`` (default 0.2): EWMA smoothing factor.
  - ``RECA_KEYPOOL_HEALTH_DUMP_EVERY_S`` (default 60) +
    ``RECA_KEYPOOL_HEALTH_DUMP`` (default 1): periodic stats printer.
"""
from __future__ import annotations

import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

from .env import env_value
from .trace import trace_event


# ── env knobs ─────────────────────────────────────────────────────────────


def _per_key_cap_for_provider(provider: str) -> int:
    raw = os.environ.get(
        f"RECA_KEYPOOL_PER_KEY_CAP_{provider.upper().replace('-', '_')}",
        "",
    ).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    raw_global = os.environ.get("RECA_KEYPOOL_PER_KEY_CAP", "").strip()
    if raw_global:
        try:
            return max(1, int(raw_global))
        except ValueError:
            pass
    return 8


def _health_dump_every() -> float:
    raw = os.environ.get("RECA_KEYPOOL_HEALTH_DUMP_EVERY_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 60.0


def _ewma_alpha() -> float:
    raw = os.environ.get("RECA_KEYPOOL_EWMA_ALPHA", "").strip()
    if raw:
        try:
            v = float(raw)
            if 0.0 < v <= 1.0:
                return v
        except ValueError:
            pass
    return 0.2   # empirically tuned


def _score_weights() -> tuple[float, float]:
    """``(w_load, w_err)`` for the scoring pick. Env-overridable."""
    def _w(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            v = float(raw)
            return max(0.0, v)
        except ValueError:
            return default
    return _w("RECA_KEYPOOL_W_LOAD", 1.0), _w("RECA_KEYPOOL_W_ERR", 1.0)


# ── cooldown policy table ────────────────────────────────────────────────
#
# Maps release ``kind`` → ``(base_cooldown_s, max_cooldown_s, escalate)``.
# Drives ``release()`` cooldown decisions per error class instead of one
# uniform exponential. Empirical
# differential-backoff rules:
#
#   - ``auth_invalid`` / ``daily_quota``: permanent (math.inf); the key
#     is gone until the process restarts or the operator rotates it.
#   - ``rate_limit`` (true 429, no Retry-After header parseable): the
#     historical kind; keep exponential 60s → 300s so adversarial
#     keys retreat fast.
#   - ``tps_throttle`` (DashScope ``AllocationQuota`` / ``qps`` /
#     ``tps``): per-second cap, recovers in seconds. **5 s fixed**,
#     no escalation — matches MEMORY note ``reference_dashscope_allocation_quota.md``.
#   - ``overload_503`` (``MODEL_CAPACITY_EXHAUSTED`` / 503): server
#     load, 10 s fixed.
#   - ``network`` / ``other``: no cooldown — let outer
#     ``retry_until_exhausted`` decide.
#
# Callers may override the policy on a per-call basis by passing
# ``cooldown_s=<float>`` to ``release()`` — this is the hook for
# Retry-After header propagation.
_COOLDOWN_POLICY: dict[str, tuple[float, float, bool]] = {
    "ok":              (0.0,       0.0,       False),
    "rate_limit":      (60.0,      300.0,     True),
    "tps_throttle":    (5.0,       15.0,      False),
    "overload_503":    (10.0,      60.0,      False),
    "daily_quota":     (math.inf,  math.inf,  False),
    "auth_invalid":    (math.inf,  math.inf,  False),
    "network":         (0.0,       0.0,       False),
    "other":           (0.0,       0.0,       False),
}


# ── KeyState ──────────────────────────────────────────────────────────────


@dataclass
class KeyState:
    """Health + concurrency tracking for a single api_key.

    EWMA tracks a sliding error rate (alpha controlled via
    ``RECA_KEYPOOL_EWMA_ALPHA``, default 0.2 — empirically tuned). A key with
    a clean run rapidly approaches 0.0; a key dropping every call asymptotes
    toward 1.0. ``pick()`` uses EWMA alongside load to score candidates so
    a degraded-but-uncooled key is naturally demoted without waiting for
    the cooldown event.
    """
    key: str
    # legacy alias kept so existing code reading ``state.api_key`` works
    api_key: str = ""
    in_flight: int = 0
    max_concurrency: int = 8
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_failure_kind: str = ""
    last_error: str | None = None
    last_error_at: float = 0.0
    total_ok: int = 0
    total_err: int = 0
    err_by_kind: dict[str, int] = field(default_factory=dict)
    # Sliding error rate ∈ [0,1]; updated on every release.
    error_rate_ewma: float = 0.0
    # Wall-clock of the last successful pick — informational.
    last_picked_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = self.key


# ── KeyPool ───────────────────────────────────────────────────────────────


@dataclass
class KeyPool:
    keys: list[KeyState] = field(default_factory=list)
    provider: str = ""
    # idle wait when every key is saturated
    pick_idle_sleep_s: float = 0.05
    # Scoring weights for pick — env-overridable, see _score_weights().
    w_load: float = field(default_factory=lambda: _score_weights()[0])
    w_err: float = field(default_factory=lambda: _score_weights()[1])
    ewma_alpha: float = field(default_factory=_ewma_alpha)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(
        cls,
        env_names: tuple[str, ...],
        default_csv_env: str | None = None,
        *,
        provider: str = "",
    ) -> "KeyPool":
        """Build a pool from a CSV env first, then legacy key env names."""
        raw_keys: list[str] = []
        if default_csv_env:
            csv = env_value(default_csv_env).strip()
            if csv:
                raw_keys.extend(k.strip() for k in csv.split(",") if k.strip())
        if not raw_keys:
            for name in env_names:
                val = env_value(name).strip()
                if val:
                    raw_keys.append(val)

        keys: list[str] = []
        seen: set[str] = set()
        for key in raw_keys:
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
        if not keys:
            raise RuntimeError(
                f"KeyPool: no API key found in {default_csv_env or ''} or {env_names}"
            )
        per_key_cap = _per_key_cap_for_provider(provider or "_")
        states = [
            KeyState(key=k, api_key=k, max_concurrency=per_key_cap)
            for k in keys
        ]
        return cls(keys=states, provider=provider)

    def pick(self) -> str:
        """Pick a healthy, under-cap key.

        Selection (simplified scoring):
          (1) Filter: ``now > cooldown_until`` AND ``in_flight < max_concurrency``
          (2) Score each candidate: ``w_load * load_ratio + w_err * error_rate_ewma``
              where ``load_ratio = in_flight / max_concurrency ∈ [0,1]``.
          (3) Pick the lowest score; ties broken randomly (avoid herding).

        When every key is cooled AND every key is at cap, degrade by
        returning the soonest-cooled key WITHOUT incrementing
        ``in_flight`` (this preserves ``max_concurrency`` as a hard cap;
        the outer retry layer will see the call fail and back off).
        A ``trace_event("keypool", "degrade_overcap", ...)`` is emitted
        so the path is observable.
        """
        while True:
            now = time.time()
            with self.lock:
                healthy = [
                    s for s in self.keys
                    if s.cooldown_until <= now and s.in_flight < s.max_concurrency
                ]
                if healthy:
                    def _score(s: KeyState) -> float:
                        load = s.in_flight / max(1, s.max_concurrency)
                        return self.w_load * load + self.w_err * s.error_rate_ewma
                    scored = [(_score(s), s) for s in healthy]
                    min_score = min(sc for sc, _ in scored)
                    candidates = [s for sc, s in scored if sc <= min_score + 1e-9]
                    chosen = random.choice(candidates)
                    chosen.in_flight += 1
                    chosen.last_picked_at = now
                    return chosen.key

                # Nothing available right now. If every key is permanently
                # disabled, bail; otherwise either wait for cooldown or for
                # a slot to free up.
                if all(s.cooldown_until == float("inf") for s in self.keys):
                    raise RuntimeError(
                        "KeyPool: every key is permanently disabled (auth_invalid)"
                    )
                cooldown_only = [
                    s for s in self.keys
                    if s.cooldown_until > now and s.cooldown_until != float("inf")
                ]
                any_free_slots = any(
                    s.in_flight < s.max_concurrency for s in self.keys
                    if s.cooldown_until != float("inf")
                )
                if cooldown_only and not any_free_slots:
                    # Degrade: return the soonest-cooled key WITHOUT
                    # incrementing in_flight (M3 fix — keeps max_concurrency
                    # a hard cap). release() tolerates the missing increment
                    # via max(0, in_flight - 1).
                    soonest = min(cooldown_only, key=lambda s: s.cooldown_until)
                    trace_event(
                        "keypool",
                        "degrade_overcap",
                        provider=self.provider,
                        key_tail=soonest.key[-6:] if len(soonest.key) > 6 else "***",
                        cooldown_remaining_s=max(0.0, soonest.cooldown_until - now),
                        in_flight=soonest.in_flight,
                        max_concurrency=soonest.max_concurrency,
                    )
                    return soonest.key
            time.sleep(self.pick_idle_sleep_s)

    def release(
        self,
        key: str,
        *,
        error_kind: str = "",
        ok: bool | None = None,
        cooldown_s: float | None = None,
    ) -> None:
        """Release a key after a provider call.

        Cooldown is decided by the ``_COOLDOWN_POLICY`` table keyed on
        ``error_kind``. Unknown kinds fall back to ``"other"``
        (no cooldown). Callers can override the table's verdict by
        passing ``cooldown_s=<float>`` — this is the hook for the
        ``Retry-After`` header propagation pattern (gateway
        ``failover_loop.go:65-125``).

        EWMA error rate is updated unconditionally (every release):
            ewma = alpha * (1 if failed else 0) + (1 - alpha) * ewma_prev

        ``error_kind`` recognized: see ``_COOLDOWN_POLICY``. Common ones:
        ``"ok"`` (success), ``"rate_limit"`` (true 429, exponential
        backoff), ``"tps_throttle"`` (DashScope AllocationQuota — short
        fixed cooldown), ``"overload_503"`` (server overload —
        10 s fixed), ``"daily_quota"`` / ``"auth_invalid"`` (permanent),
        ``"network"`` / ``"other"`` (no cooldown — let outer retry
        decide).

        ``ok=True`` overrides ``error_kind`` to record a success;
        ``ok=False`` defaults to ``"other"`` if no error_kind given.
        """
        if ok is True:
            kind = "ok"
        elif ok is False and not error_kind:
            kind = "other"
        else:
            kind = error_kind or "ok"

        failed = (kind != "ok")
        policy = _COOLDOWN_POLICY.get(kind, _COOLDOWN_POLICY["other"])
        base_cd, cap_cd, escalate = policy
        alpha = self.ewma_alpha

        with self.lock:
            for state in self.keys:
                if state.key != key:
                    continue
                state.in_flight = max(0, state.in_flight - 1)
                # EWMA on every release — applies to both ok and failed.
                state.error_rate_ewma = (
                    alpha * (1.0 if failed else 0.0)
                    + (1.0 - alpha) * state.error_rate_ewma
                )
                if kind == "ok":
                    state.consecutive_failures = 0
                    state.last_failure_kind = ""
                    state.total_ok += 1
                    return
                # Failure path: bookkeeping
                state.total_err += 1
                state.err_by_kind[kind] = state.err_by_kind.get(kind, 0) + 1
                state.last_failure_kind = kind
                state.consecutive_failures += 1
                # Cooldown decision: caller override > policy table.
                if cooldown_s is not None:
                    cd = max(0.0, float(cooldown_s))
                elif math.isinf(base_cd):
                    cd = math.inf
                elif escalate:
                    cd = min(
                        base_cd * (2 ** (state.consecutive_failures - 1)),
                        cap_cd,
                    )
                else:
                    cd = min(base_cd, cap_cd)
                if math.isinf(cd):
                    state.cooldown_until = float("inf")
                elif cd > 0:
                    state.cooldown_until = time.time() + cd
                # cd == 0: no cooldown set (network/other) — let outer retry decide.
                return

    def stats(self) -> list[dict]:
        now = time.time()
        with self.lock:
            return [
                {
                    "key_tail": s.key[-6:] if len(s.key) > 6 else "***",
                    "in_flight": s.in_flight,
                    "max_concurrency": s.max_concurrency,
                    "cooldown_remaining_s": (
                        float("inf") if math.isinf(s.cooldown_until)
                        else max(0.0, s.cooldown_until - now)
                    ),
                    "consecutive_failures": s.consecutive_failures,
                    "last_failure_kind": s.last_failure_kind,
                    "total_ok": s.total_ok,
                    "total_err": s.total_err,
                    "err_by_kind": dict(s.err_by_kind),
                    "error_rate_ewma": s.error_rate_ewma,
                }
                for s in self.keys
            ]


# ── pool cache + health dump ──────────────────────────────────────────────


_POOLS: dict[str, KeyPool] = {}
_POOLS_LOCK = threading.Lock()
_HEALTH_THREAD_STARTED = False
_HEALTH_THREAD_LOCK = threading.Lock()


def cached_key_pool(
    name: str,
    env_names: tuple[str, ...],
    *,
    default_csv_env: str | None = None,
) -> KeyPool:
    """Return one process-local pool per logical provider name."""
    cache_key = name
    with _POOLS_LOCK:
        pool = _POOLS.get(cache_key)
        if pool is None:
            pool = KeyPool.from_env(
                env_names, default_csv_env=default_csv_env, provider=name,
            )
            _POOLS[cache_key] = pool
            _maybe_start_health_dumper()
        return pool


def _maybe_start_health_dumper() -> None:
    """Start a single daemon thread that prints pool health periodically."""
    global _HEALTH_THREAD_STARTED
    if _HEALTH_THREAD_STARTED:
        return
    every = _health_dump_every()
    if every <= 0:
        return
    if os.environ.get("RECA_KEYPOOL_HEALTH_DUMP", "1").strip() in ("0", "false", "no"):
        return
    with _HEALTH_THREAD_LOCK:
        if _HEALTH_THREAD_STARTED:
            return
        _HEALTH_THREAD_STARTED = True
        threading.Thread(
            target=_health_dump_loop,
            args=(every,),
            name="keypool-health",
            daemon=True,
        ).start()


def _health_dump_loop(period_s: float) -> None:
    while True:
        try:
            time.sleep(period_s)
            with _POOLS_LOCK:
                snapshot = {name: pool.stats() for name, pool in _POOLS.items()}
            for name, stats in snapshot.items():
                in_flights = [s["in_flight"] for s in stats]
                cds = [
                    "yes" if s["cooldown_remaining_s"] > 0 else "no"
                    for s in stats
                ]
                err_kinds_union: dict[str, int] = {}
                for s in stats:
                    for k, v in s.get("err_by_kind", {}).items():
                        err_kinds_union[k] = err_kinds_union.get(k, 0) + v
                print(
                    f"[keypool] {name}: keys={len(stats)} "
                    f"in_flight={in_flights} cooldown={cds} "
                    f"err={err_kinds_union}",
                    flush=True,
                )
        except Exception:  # noqa: BLE001
            # never let the daemon thread kill itself
            continue


# ── error classifier ──────────────────────────────────────────────────────


def classify_error(exc: BaseException) -> str:
    """Map SDK / HTTP exceptions to KeyPool release kinds.

    Order matters — check more-specific markers first. The output kind
    is consumed by ``release()`` to pick a cooldown from
    ``_COOLDOWN_POLICY``; misclassification translates directly to
    wrong-duration cooldowns.

    Kind taxonomy mirrors a typical gateway failover-loop classification:
      - ``auth_invalid``: 401 / 403 / "invalid api key" — permanent
      - ``daily_quota``: DailyQuota / MonthlyQuota — permanent
      - ``tps_throttle``: AllocationQuota / qps / tps / tpm — short fixed
      - ``rate_limit``: 429 / Throttling / RateLimit / RateQuota — exponential
      - ``overload_503``: ModelCapacity / overload / 503 — short fixed
      - ``network``: timeout / connection / 5xx — no cooldown
      - ``other``: catch-all — no cooldown
    """
    msg = str(exc).lower()
    # auth: check before rate_limit so "Unauthorized: rate exceeded" doesn't
    # mis-classify (real strings can mention both).
    if any(s in msg for s in ("invalidapikey", "unauthor", "forbidden",
                              "401", "403")):
        return "auth_invalid"
    if any(s in msg for s in ("dailyquota", "monthlyquota")):
        return "daily_quota"
    # TPS / per-second rate cap: distinct from true daily quota in name
    # but DashScope's misleading "AllocationQuota" message belongs here
    # (see MEMORY: reference_dashscope_allocation_quota.md).
    if any(s in msg for s in ("allocationquota", "qps", "tps", "tpm")):
        return "tps_throttle"
    # True 429 / rate-limit: keep the exponential-backoff kind name.
    if any(s in msg for s in ("ratequota", "ratelimit", "throttling",
                              "rate limit", "rate_limit", "429",
                              "too many requests")):
        return "rate_limit"
    if any(s in msg for s in ("modelcapacity", "model_capacity_exhausted",
                              "overload", "503")):
        return "overload_503"
    if any(s in msg for s in ("timeout", "timed out", "connection",
                              "econnreset", "network", "502", "504")):
        return "network"
    return "other"
