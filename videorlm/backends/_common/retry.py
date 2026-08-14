"""Same-backend transient-error retry loop.

Replaces the now-deleted ``fallback.py`` infrastructure. Single rule: this
module only retries the same backend; it MUST NOT route to a different
backend / model / provider. Cross-backend fallback violates the design
principle in ``docs/backends.md``.

Public API:
    RetryPolicy                — frozen dataclass of retry knobs
    retry_until_exhausted(...) — call_fn(), retry on transient errors until
                                  policy.max_attempts is exhausted
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from .trace import trace_event

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Per-backend retry policy.

    All retries are SAME-backend / SAME-model. Cross-backend chaining is
    forbidden by ``docs/backends.md §8``.
    """

    max_attempts: int = 6
    backoff_base_s: float = 1.5
    backoff_cap_s: float = 60.0
    transient_errors: frozenset[type[BaseException]] = field(default_factory=frozenset)
    transient_msg_substrings: frozenset[str] = field(default_factory=frozenset)


def _is_transient(exc: BaseException, policy: RetryPolicy) -> bool:
    if any(isinstance(exc, cls) for cls in policy.transient_errors):
        return True
    msg = f"{type(exc).__name__} {str(exc)}".lower()
    return any(marker.lower() in msg for marker in policy.transient_msg_substrings)


def retry_until_exhausted(
    backend_name: str,
    call_fn: Callable[[], T],
    *,
    policy: RetryPolicy,
) -> T:
    """Retry ``call_fn`` on the SAME backend until ``policy.max_attempts`` exhausted.

    NO cross-backend fallback. NO chain. NO tier. Caller is responsible for
    selecting backend; this function only handles transient-error retry
    with exponential backoff.

    Returns the call_fn result on success. On exhaustion, raises a
    ``RuntimeError`` with the last error summary.
    """
    max_attempts = max(1, int(policy.max_attempts))
    last_err: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        trace_event(
            "retry", "attempt_start",
            backend=backend_name,
            attempt=attempt, max_attempts=max_attempts,
        )
        try:
            result = call_fn()
            trace_event(
                "retry", "success",
                backend=backend_name,
                attempt=attempt, max_attempts=max_attempts,
            )
            if attempt > 1:
                logger.info(
                    "[retry] backend=%s succeeded after retry %d",
                    backend_name, attempt,
                )
            return result
        except BaseException as exc:
            if not _is_transient(exc, policy):
                trace_event(
                    "retry", "non_transient_error",
                    backend=backend_name,
                    attempt=attempt, max_attempts=max_attempts,
                    error_type=type(exc).__name__, error=str(exc)[:500],
                )
                raise
            last_err = exc
            if attempt >= max_attempts:
                trace_event(
                    "retry", "exhausted",
                    backend=backend_name,
                    attempt=attempt, max_attempts=max_attempts,
                    error_type=type(exc).__name__, error=str(exc)[:500],
                )
                logger.info(
                    "[retry] backend=%s transient-exhausted after %d attempts",
                    backend_name, max_attempts,
                )
                break
            delay = min(
                float(policy.backoff_cap_s),
                float(policy.backoff_base_s) * (2 ** (attempt - 1)),
            )
            logger.info(
                "[retry] backend=%s transient %s attempt %d/%d; backoff %.1fs",
                backend_name, type(exc).__name__, attempt, max_attempts, delay,
            )
            trace_event(
                "retry", "transient_backoff",
                backend=backend_name,
                attempt=attempt, max_attempts=max_attempts,
                delay_s=delay, error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            time.sleep(delay)
    raise RuntimeError(
        f"backend {backend_name} retry exhausted "
        f"(max_attempts={max_attempts}; last: "
        f"{type(last_err).__name__}: {str(last_err)[:200]})"
    )
