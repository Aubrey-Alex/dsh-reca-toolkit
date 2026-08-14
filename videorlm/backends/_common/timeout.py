"""Small timeout wrapper for provider calls.

This is a guardrail against SDK calls that block forever in socket read. The
wrapped worker cannot be killed by Python, but the caller regains control and
can route the failure into the retry chain.
"""
from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

from .trace import trace_event

T = TypeVar("T")


class BackendCallTimeout(TimeoutError):
    """A backend call exceeded its wall-clock timeout."""


@dataclass
class _TimeoutRunner:
    max_workers: int
    executor: concurrent.futures.ThreadPoolExecutor
    semaphore: threading.BoundedSemaphore


_RUNNER_LOCK = threading.Lock()
_RUNNER: _TimeoutRunner | None = None


def default_timeout_s(env_name: str, fallback_s: float) -> float:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return fallback_s
    try:
        return max(0.0, float(raw))
    except ValueError:
        return fallback_s


def _default_timeout_workers() -> int:
    raw = os.environ.get("RECA_BACKEND_TIMEOUT_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    # Covers the expected 16-way W1/W2 fanout plus concurrent validators/video
    # submits, while still bounding zombie SDK threads after socket hangs.
    return 64


def _slot_wait_s(timeout_s: float) -> float:
    raw = os.environ.get("RECA_BACKEND_TIMEOUT_SLOT_WAIT_S", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return min(1.0, max(0.0, timeout_s))


def _runner() -> _TimeoutRunner:
    global _RUNNER
    workers = _default_timeout_workers()
    with _RUNNER_LOCK:
        if _RUNNER is None or _RUNNER.max_workers != workers:
            old = _RUNNER
            _RUNNER = _TimeoutRunner(
                max_workers=workers,
                executor=concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="backend-timeout",
                ),
                semaphore=threading.BoundedSemaphore(workers),
            )
            if old is not None:
                old.executor.shutdown(wait=False, cancel_futures=True)
        return _RUNNER


def _reset_timeout_runner_for_tests() -> None:
    global _RUNNER
    with _RUNNER_LOCK:
        old = _RUNNER
        _RUNNER = None
    if old is not None:
        old.executor.shutdown(wait=False, cancel_futures=True)


def _run_and_release(fn: Callable[[], T], semaphore: threading.BoundedSemaphore) -> T:
    try:
        return fn()
    finally:
        semaphore.release()


def call_with_timeout(
    fn: Callable[[], T],
    *,
    timeout_s: float,
    label: str,
) -> T:
    """Run ``fn`` with a wall-clock timeout.

    ``timeout_s <= 0`` disables the wrapper.
    """
    if timeout_s <= 0:
        return fn()

    runner = _runner()
    trace_event(
        "timeout", "slot_wait_start",
        label=label, timeout_s=timeout_s, workers=runner.max_workers,
    )
    acquired = runner.semaphore.acquire(timeout=_slot_wait_s(timeout_s))
    if not acquired:
        trace_event(
            "timeout", "slot_wait_timeout",
            label=label, timeout_s=timeout_s, workers=runner.max_workers,
        )
        raise BackendCallTimeout(
            f"{label} timed out waiting for timeout-worker slot "
            f"(workers={runner.max_workers})"
        )

    t0 = time.time()
    trace_event(
        "timeout", "call_start",
        label=label, timeout_s=timeout_s, workers=runner.max_workers,
    )
    fut = runner.executor.submit(_run_and_release, fn, runner.semaphore)
    try:
        result = fut.result(timeout=timeout_s)
        trace_event(
            "timeout", "call_success",
            label=label, timeout_s=timeout_s,
            elapsed_s=time.time() - t0,
        )
        return result
    except concurrent.futures.TimeoutError as exc:
        cancelled = fut.cancel()
        if cancelled:
            runner.semaphore.release()
        trace_event(
            "timeout", "call_timeout",
            label=label, timeout_s=timeout_s,
            elapsed_s=time.time() - t0,
            cancelled=cancelled,
        )
        raise BackendCallTimeout(f"{label} timed out after {timeout_s:.1f}s") from exc
    except Exception as exc:
        trace_event(
            "timeout", "call_error",
            label=label, timeout_s=timeout_s,
            elapsed_s=time.time() - t0,
            error_type=type(exc).__name__, error=str(exc)[:500],
        )
        raise
