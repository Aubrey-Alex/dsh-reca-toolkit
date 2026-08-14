"""Tests for ``backends._common.concurrency.NamedSemaphorePool``."""
from __future__ import annotations

import threading
import time

import pytest

from videorlm.backends._common.concurrency import NamedSemaphorePool


def test_semaphore_for_caches_by_name():
    pool = NamedSemaphorePool()
    a1 = pool.semaphore_for("wan", 4)
    a2 = pool.semaphore_for("wan", 99)   # different capacity arg
    assert a1 is a2, "second call must return cached semaphore"


def test_semaphore_for_distinct_names_returns_distinct_objects():
    pool = NamedSemaphorePool()
    a = pool.semaphore_for("wan", 4)
    b = pool.semaphore_for("happyhorse", 4)
    assert a is not b


def test_capacity_locked_at_first_construction():
    """Second call shouldn't shrink (or grow) the existing semaphore."""
    pool = NamedSemaphorePool()
    pool.semaphore_for("x", 2)
    # Acquire both of the 2 slots.
    sem = pool.semaphore_for("x", 99)   # passing big capacity must NOT widen
    sem.acquire(); sem.acquire()
    assert not sem.acquire(blocking=False), \
        "3rd acquire must block — capacity stayed at 2"
    sem.release(); sem.release()


def test_env_override_beats_caller_capacity(monkeypatch):
    monkeypatch.setenv("RECA_TEST_CONCURRENCY_WAN_2_7_I2V", "3")
    pool = NamedSemaphorePool(env_prefix="RECA_TEST_CONCURRENCY_")
    sem = pool.semaphore_for("wan-2.7-i2v", 99)
    # Should have 3 permits even though caller said 99.
    for _ in range(3):
        assert sem.acquire(blocking=False)
    assert not sem.acquire(blocking=False), \
        f"env override should cap at 3, got more"
    for _ in range(3):
        sem.release()


def test_env_override_falls_back_to_caller_when_env_absent(monkeypatch):
    pool = NamedSemaphorePool(env_prefix="RECA_TEST_NEVER_SET_")
    sem = pool.semaphore_for("anything", 2)
    assert sem.acquire(blocking=False)
    assert sem.acquire(blocking=False)
    assert not sem.acquire(blocking=False), "capacity must be 2"
    sem.release(); sem.release()


def test_env_override_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("RECA_TEST_CONCURRENCY_X", "not-a-number")
    pool = NamedSemaphorePool(env_prefix="RECA_TEST_CONCURRENCY_")
    sem = pool.semaphore_for("x", 5)
    permits = sum(1 for _ in range(10) if sem.acquire(blocking=False))
    assert permits == 5, f"invalid env → fallback to caller=5, got {permits}"
    for _ in range(permits):
        sem.release()


def test_slot_context_manager_acquires_and_releases():
    pool = NamedSemaphorePool()
    sem = pool.semaphore_for("y", 1)
    with pool.slot("y", 1):
        # Inside the with-block, no more slots available.
        assert not sem.acquire(blocking=False), "slot didn't acquire"
    # After the with-block, slot was released.
    assert sem.acquire(blocking=False)
    sem.release()


def test_slot_releases_on_exception():
    pool = NamedSemaphorePool()
    sem = pool.semaphore_for("z", 1)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with pool.slot("z", 1):
            raise Boom()

    # Slot must be back.
    assert sem.acquire(blocking=False), "slot leaked on exception"
    sem.release()


def test_bounded_semaphore_catches_over_release():
    """Using BoundedSemaphore (not plain Semaphore) means over-release
    raises immediately — surfaces bugs instead of silently inflating cap."""
    pool = NamedSemaphorePool()
    sem = pool.semaphore_for("over-release-test", 2)
    sem.acquire()
    sem.release()
    with pytest.raises(ValueError):
        sem.release()  # over-release: BoundedSemaphore catches this


def test_concurrent_slot_acquisition_respects_cap():
    """20 threads racing for a 3-permit pool — peak concurrency must be ≤3."""
    pool = NamedSemaphorePool()
    peak = [0]
    cur = [0]
    cur_lock = threading.Lock()

    def _worker():
        with pool.slot("hot", 3):
            with cur_lock:
                cur[0] += 1
                if cur[0] > peak[0]:
                    peak[0] = cur[0]
            time.sleep(0.01)
            with cur_lock:
                cur[0] -= 1

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak[0] <= 3, f"breached cap: peak={peak[0]}"
    assert peak[0] >= 2, f"middleware never held >1 slot (peak={peak[0]})"


def test_reset_for_tests_clears_registry():
    pool = NamedSemaphorePool()
    pool.semaphore_for("foo", 1)
    pool.semaphore_for("bar", 1)
    assert pool.names() == ["bar", "foo"]
    pool.reset_for_tests()
    assert pool.names() == []


def test_default_no_env_prefix_ignores_env(monkeypatch):
    """Without env_prefix, env variables shouldn't influence the pool."""
    monkeypatch.setenv("RECA_BACKEND_CONCURRENCY_FOO", "99")
    pool = NamedSemaphorePool()   # no env_prefix
    sem = pool.semaphore_for("foo", 2)
    permits = sum(1 for _ in range(10) if sem.acquire(blocking=False))
    assert permits == 2, f"env should be ignored without env_prefix, got {permits}"
    for _ in range(permits):
        sem.release()
