"""DashScope SDK helpers — internal, self-contained.

Inlined from the former ``videorlm/shared_dashscope/utils.py`` and
``videorlm/shared_dashscope/qwen_client.py`` so the ``backends/`` package
is truly standalone (no reverse-imports into outer subpackages). The
prior shim doc claimed "backends/ → _common/dashscope_sdk → shared_dashscope"
but that broke README §0.1's standalone claim. We now own the code here.

Symbol map (all the dashscope-SDK touchpoints used by ``backends/``):

  Constants:     ``DEFAULT_RES`` ``NEG`` ``WAN_I2V`` ``WANX_T2I``
  Logging:       ``log_wan_call``
  Submit/Retry:  ``submit_with_retry`` (+ ``RateLimiter`` / per-thread
                 stats / decorrelated-jitter backoff)
  Poll:          ``poll_task`` ``poll_image_task``
  Download:      ``download_file``
  Qwen:          ``strip_think_tags``
  Response:      ``classify_dashscope_response``

The DashScope async-task retry loop here is the historical, battle-tested
implementation copied verbatim from the Stage1 baseline runner. It runs
INSIDE a single ``with_key("dashscope", ...)`` envelope per attempt
(see ``media/impl/dashscope/_platform.submit_with_key``), so the two
concerns layer cleanly:

  - **submit_with_retry**:  outer retry loop, content-filter aware,
                             per-model min-interval pacing via RateLimiter
  - **with_key middleware**: inner per-attempt key pick + release with
                             error_kind classification
"""
from __future__ import annotations

import json as _json
import os
import random
import re
import threading
import time
from http import HTTPStatus

import httpx

from .platforms import get_platform


# ─── Constants ──────────────────────────────────────────────────────────────


WAN_T2V = "wan2.7-t2v"
WAN_I2V = "wan2.7-i2v"
WANX_T2I = "wanx-v1"
DEFAULT_RES = "1280*720"
NEG = (
    "blurry, low quality, distorted face, distorted body, different person, "
    "changed clothing, inconsistent wardrobe, text, text overlay, subtitles, "
    "logo, watermark"
)


# ─── API key resolution (lazy, delegates to platforms.get_platform) ────────


def _resolve_api_key() -> str:
    return get_platform("dashscope").api_key()


def _ensure_api_key() -> str:
    key = _resolve_api_key()
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. Export DASHSCOPE_API_KEY (or "
            "DASHSCOPE_API_KEYS=k1,k2,... for KeyPool fan-out)."
        )
    return key


# ─── Wan call logging ───────────────────────────────────────────────────────


_PROMPT_LOG_LOCK = threading.Lock()
_PROMPT_CALL_COUNT = 0


def log_wan_call(
    mode: str,
    model: str,
    prompt: str,
    ref_images: list | None = None,
    seed: int = 0,
    extra: dict | None = None,
    log_dir: str | None = None,
) -> None:
    """Log a Wan API call to stdout + optional per-cell JSONL file."""
    global _PROMPT_CALL_COUNT
    with _PROMPT_LOG_LOCK:
        _PROMPT_CALL_COUNT += 1
        idx = _PROMPT_CALL_COUNT
    entry = {
        "call_idx": idx,
        "mode": mode,
        "model": model,
        "prompt": prompt,
        "prompt_words": len(prompt.split()),
        "ref_images": ref_images or [],
        "seed": seed,
        **(extra or {}),
    }
    print(
        f"  [WAN {mode.upper()} #{idx}] {entry['prompt_words']}w prompt: {prompt}",
        flush=True,
    )
    if log_dir:
        log_path = os.path.join(log_dir, "wan_calls.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        with _PROMPT_LOG_LOCK:
            with open(log_path, "a") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Thread-local stats counters ───────────────────────────────────────────
#
# Each cell (= ThreadPoolExecutor worker) keeps its own counters; no
# cross-thread contamination. Tickers print these every 60 s in the driver
# (run_baselines.py uses get_stats() to dump them).


_THREAD_LOCAL = threading.local()
_STATS_KEYS = (
    "submit_ok", "submit_throttle_retry", "submit_transient_retry",
    "submit_hard_fail", "content_filter_blocked", "ip_infringement",
    "poll_timeout", "total_retries",
)


def _ensure_thread_stats() -> None:
    if not hasattr(_THREAD_LOCAL, "stats"):
        _THREAD_LOCAL.stats = {k: 0 for k in _STATS_KEYS}


def get_stats() -> dict:
    _ensure_thread_stats()
    return dict(_THREAD_LOCAL.stats)


def reset_stats() -> None:
    _THREAD_LOCAL.stats = {k: 0 for k in _STATS_KEYS}


def _stat(key: str, n: int = 1) -> None:
    _ensure_thread_stats()
    _THREAD_LOCAL.stats[key] = _THREAD_LOCAL.stats.get(key, 0) + n


# ─── Retry / pacing env knobs ──────────────────────────────────────────────


MAX_SUBMIT_RETRIES = int(os.environ.get("MAX_SUBMIT_RETRIES", "16"))
MAX_CONTENT_RETRIES = int(os.environ.get("MAX_CONTENT_RETRIES", "2"))
MAX_CELL_RESUBMITS = int(os.environ.get("MAX_CELL_RESUBMITS", "3"))
POLL_MAX_WAIT_S = int(os.environ.get("POLL_MAX_WAIT_S", "3600"))
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "8"))

# Global fallback for unknown models. Hardcoded — env vars are reserved
# for provider selection + API keys; rate-limit values come from JSON.
# Re-exported for back-compat with callers reading ``ds.SUBMIT_*``.
from . import rate_limiter as _rl

SUBMIT_MIN_INTERVAL_S = _rl.DEFAULT_INTERVAL_S
SUBMIT_MAX_PARALLEL_DEFAULT = _rl.DEFAULT_MAX_PARALLEL


# ─── Per-model rate-limit table ────────────────────────────────────────────
#
# Single source: ``configs/dashscope_rate_limits.json``. Swap-only-the-file
# env override: ``DASHSCOPE_RATE_LIMITS_CONFIG=<path>``. There are NO
# per-value env overrides — to change a model's interval / max_parallel,
# edit the JSON.
#
# Schema (in-memory): {model_name: {"interval": float_seconds, "max_parallel": int}}
# Schema (JSON file): {"models": {"<model>": {"interval_s": float, "max_parallel": int}, ...}}


# Provider-agnostic ``RateLimiter`` lives in ``_common/rate_limiter.py``;
# the dashscope provider's ``rate_limits()`` returns the JSON-derived
# table. ``cached_rate_limiter("dashscope")`` is the process-wide singleton.
from .providers import get_provider
from .rate_limiter import RateLimiter, cached_rate_limiter


def _load_rate_limits_json() -> dict[str, dict[str, float | int]]:
    """Back-compat shim — returns the dashscope provider's resolved
    rate-limits table (loaded from the bundled JSON)."""
    return get_provider("dashscope").rate_limits()


_MODEL_LIMITS = _load_rate_limits_json()


def reload_model_limits() -> dict[str, dict[str, float | int]]:
    """Re-snapshot ``_MODEL_LIMITS`` from the dashscope provider. Useful
    after env changes since ``Provider.rate_limits()`` is called fresh
    each time but ``_MODEL_LIMITS`` is a module-level cache."""
    global _MODEL_LIMITS
    _MODEL_LIMITS = _load_rate_limits_json()
    return _MODEL_LIMITS


_RATE = cached_rate_limiter("dashscope")


def _is_throttle(code, msg) -> bool:
    c = str(code or "")
    m = str(msg or "")
    return (
        "Throttl" in c or "RateQuota" in c or "429" in c or "429" in m
        or "quota" in m.lower() or "rate limit" in m.lower()
    )


def _is_transient(status_code, code, msg) -> bool:
    if status_code is not None and 500 <= int(status_code) < 600:
        return True
    c = str(code or "").lower()
    m = str(msg or "").lower()
    markers = (
        "internal", "timeout", "timed out", "gateway", "unavailable",
        "connection reset", "try again", "retry", "overload",
    )
    return any(t in c or t in m for t in markers)


def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 60.0) -> None:
    """Decorrelated-jitter backoff."""
    time.sleep(min(cap, base * (2 ** min(attempt, 6)) + random.uniform(0, 2)))


# ─── Submit / poll / download ──────────────────────────────────────────────


def submit_with_retry(call_fn, label: str, rate_key: str) -> str:
    """Submit a DashScope async task with retry. Returns ``task_id``.

    ``call_fn``: callable returning an SDK response object with
        ``.status_code`` / ``.code`` / ``.message`` / ``.output.task_id``.
        Typically wrapped via ``submit_with_key(lambda api_key: ...)`` so
        per-attempt key rotation happens inside.
    ``rate_key``: model name for ``RateLimiter`` lookup (e.g. ``"wan2.7-i2v"``).
    """
    last = None
    for attempt in range(MAX_SUBMIT_RETRIES):
        gate = _RATE.acquire(rate_key)
        try:
            try:
                rsp = call_fn()
            except Exception as e:  # noqa: BLE001
                last = f"EXC {type(e).__name__}: {str(e)[:160]}"
                _stat("total_retries")
                backoff_sleep(attempt)
                continue
        finally:
            _RATE.release(gate)
        if rsp.status_code == HTTPStatus.OK:
            _stat("submit_ok")
            return rsp.output.task_id
        code = getattr(rsp, "code", "") or ""
        msg = (getattr(rsp, "message", "") or "")[:200]
        status = getattr(rsp, "status_code", None)
        last = f"status={status} code={code} msg={msg}"
        if code == "DataInspectionFailed":
            _stat("content_filter_blocked")
        if code == "IPInfringementSuspect":
            _stat("ip_infringement")
        if _is_throttle(code, msg):
            _stat("submit_throttle_retry")
            _stat("total_retries")
            backoff_sleep(attempt)
            continue
        if _is_transient(status, code, msg):
            _stat("submit_transient_retry")
            _stat("total_retries")
            backoff_sleep(attempt)
            continue
        break
    _stat("submit_hard_fail")
    raise RuntimeError(f"{label} submit failed: {last}")


def poll_task(task_id: str, max_wait: int = POLL_MAX_WAIT_S) -> dict:
    """Poll a DashScope ``VideoSynthesis`` task until terminal."""
    api_key = _ensure_api_key()
    from dashscope import VideoSynthesis  # lazy: avoid SDK at import time

    t0 = time.time()
    n_exc = 0
    while time.time() - t0 < max_wait:
        try:
            rsp = VideoSynthesis.fetch(task=task_id, api_key=api_key)
            out = rsp.output
            d = out if isinstance(out, dict) else {
                k: getattr(out, k, None) for k in
                ["task_id", "task_status", "video_url", "code", "message"]
            }
            if d.get("task_status") in ("SUCCEEDED", "FAILED", "UNKNOWN"):
                return d
        except Exception:  # noqa: BLE001
            n_exc += 1
            time.sleep(min(30, POLL_INTERVAL_S + n_exc))
            continue
        time.sleep(POLL_INTERVAL_S)
    return {"task_status": "TIMEOUT", "task_id": task_id}


def poll_image_task(task_id: str, max_wait: int = 600) -> dict:
    """Poll a DashScope ``ImageSynthesis`` task until terminal."""
    api_key = _ensure_api_key()
    from dashscope import ImageSynthesis  # lazy

    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            rsp = ImageSynthesis.fetch(task=task_id, api_key=api_key)
            out = rsp.output
            status = (
                out.get("task_status") if isinstance(out, dict)
                else getattr(out, "task_status", None)
            )
            if status in ("SUCCEEDED", "FAILED", "UNKNOWN"):
                if isinstance(out, dict):
                    return out
                return {
                    k: getattr(out, k, None) for k in
                    ["task_id", "task_status", "results", "code", "message"]
                }
        except Exception:  # noqa: BLE001
            time.sleep(POLL_INTERVAL_S)
            continue
        time.sleep(POLL_INTERVAL_S)
    return {"task_status": "TIMEOUT", "task_id": task_id}


def download_file(url: str, path: str, max_retries: int = 5) -> int:
    """Download URL to local path with retry. Returns file size in bytes."""
    for attempt in range(max_retries):
        try:
            with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes(8192):
                        f.write(chunk)
            return os.path.getsize(path)
        except Exception:  # noqa: BLE001
            if attempt < max_retries - 1:
                backoff_sleep(attempt, base=2.0)
    raise RuntimeError(f"Download failed after {max_retries} retries: {url}")


# ─── Qwen response post-processing ─────────────────────────────────────────


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from Qwen 3.x responses.

    Qwen 3.6-plus prepends reasoning in ``<think>`` tags by default; this
    breaks JSON parsing downstream when callers expect raw JSON.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ─── Response classifier (consumed by with_key middleware) ─────────────────
# Single source of truth lives in providers/dashscope.py so the provider
# itself wires it as response_classifier. Re-exported here for callers
# that import via the dashscope_sdk surface.
from .providers.dashscope import classify_dashscope_response  # noqa: E402,F401


__all__ = [
    # Constants / model names
    "DEFAULT_RES",
    "NEG",
    "WAN_I2V",
    "WAN_T2V",
    "WANX_T2I",
    # Logging
    "log_wan_call",
    # Submit / poll / download
    "submit_with_retry",
    "poll_task",
    "poll_image_task",
    "download_file",
    "backoff_sleep",
    "RateLimiter",
    "reload_model_limits",
    # Stats
    "get_stats",
    "reset_stats",
    # Qwen post-processing
    "strip_think_tags",
    # Response classifier (for with_key middleware)
    "classify_dashscope_response",
]
