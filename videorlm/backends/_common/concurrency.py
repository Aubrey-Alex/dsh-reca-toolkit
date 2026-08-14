"""Per-name ``BoundedSemaphore`` registry for backend concurrency caps.

Used by both ``llm/agents/base.py`` (per-agent slot semaphore) and
``media/interface/dispatch.py`` (per-backend dispatch semaphore) so the
"per-name concurrency cap" pattern lives in one place instead of being
re-implemented under two slightly different APIs.

Construction:

    # No env override — capacity strictly from caller.
    pool = NamedSemaphorePool()

    # Env override:  ``RECA_BACKEND_CONCURRENCY_<NAME>``  trumps the
    # caller's capacity. NAME is uppercased, ``-`` / ``.`` → ``_``.
    pool = NamedSemaphorePool(env_prefix="RECA_BACKEND_CONCURRENCY_")

Acquire/release via the context manager (or the raw semaphore object
when a context manager isn't ergonomic):

    with pool.slot("wan2.7-i2v", capacity=8):
        ... call the backend ...

    sem = pool.semaphore_for("wan2.7-i2v", 8)
    sem.acquire(); try: ...; finally: sem.release()

Once a name's semaphore is created, subsequent ``semaphore_for(name, X)``
calls return the SAME semaphore regardless of the new ``capacity``
argument — capacity is locked at first construction. This prevents
silent shrinking by accident.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Iterator


class NamedSemaphorePool:
    """Thread-safe registry of per-name ``BoundedSemaphore``.

    ``BoundedSemaphore`` is preferred over the unbounded ``Semaphore`` so
    over-release bugs raise ``ValueError`` instead of silently inflating
    the cap.
    """

    __slots__ = ("_lock", "_semaphores", "_env_prefix")

    def __init__(self, env_prefix: str | None = None) -> None:
        self._lock = threading.Lock()
        self._semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._env_prefix = env_prefix or None

    def _resolved_capacity(self, name: str, capacity: int) -> int:
        if self._env_prefix:
            env_key = (
                self._env_prefix
                + name.upper().replace(".", "_").replace("-", "_")
            )
            raw = os.environ.get(env_key, "").strip()
            if raw:
                try:
                    return max(1, int(raw))
                except ValueError:
                    pass
        return max(1, int(capacity))

    def semaphore_for(
        self, name: str, capacity: int,
    ) -> threading.BoundedSemaphore:
        """Return the (cached) ``BoundedSemaphore`` for ``name``.

        Creates one on first call using the env-override-or-``capacity``
        rule. Subsequent calls ignore ``capacity`` and return the cached
        instance.
        """
        with self._lock:
            sem = self._semaphores.get(name)
            if sem is None:
                effective = self._resolved_capacity(name, capacity)
                sem = threading.BoundedSemaphore(effective)
                self._semaphores[name] = sem
            return sem

    @contextmanager
    def slot(self, name: str, capacity: int) -> Iterator[None]:
        """Block on the semaphore for ``name``; release in ``finally``."""
        sem = self.semaphore_for(name, capacity)
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._semaphores.keys())

    def reset_for_tests(self) -> None:
        """Clear the registry. ONLY for test isolation — never call in
        production code (live in-flight callers will lose their
        semaphore on next ``semaphore_for`` if they re-resolve)."""
        with self._lock:
            self._semaphores.clear()


__all__ = ["NamedSemaphorePool"]
