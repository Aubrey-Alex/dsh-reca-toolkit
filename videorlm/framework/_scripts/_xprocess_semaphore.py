"""Cross-process semaphore wrapper for video / agent dispatch.

This module is a pure wrapper. It does NOT modify any code in
``videorlm/backends/``. Backend code is untouched and works exactly as
before in single-process mode.

Two scopes are gated:
- ``videorlm.backends.media`` dispatch_segment / dispatch_bridge / dispatch_image
  → user-level total concurrency capped per backend across processes.
- ``videorlm.backends.llm.agents`` OpenAICompatibleAgent.prompt
  (covers ``DashScopeQwenAgent`` via MRO inheritance)
  → gpt-5.5 / qwen* chat concurrency capped across processes.

Implementation: filesystem-based semaphore via ``fcntl.flock`` + per-slot
marker files in ``/tmp/videorlm_xsem/<backend>/``. Stale slots (older
than TTL) are auto-collected — kill -9 of a process won't permanently
hold slots.

Single-process / unchanged behavior:
    Do NOT call enable_xsem(). Everything works as before.

Multi-process / cross-process limit:
    At process start (BEFORE importing videorlm.framework.pipeline):

        from videorlm.framework._xprocess_semaphore import enable_xsem
        enable_xsem()

    Or via env var ``RECA_XSEM_ENABLE=1``, in which case _smoke.py
    auto-enables.

Configuration (defaults match observed user-level server quotas; tune
via env var or pass dict to enable_xsem):

    RECA_XSEM_HAPPYHORSE_R2V    = 18    # server quota ~20-30
    RECA_XSEM_GPT_IMAGE_2_PRO   = 12    # gateway user-level ~15-20
    RECA_XSEM_GPT_5_5           = 12    # same gateway
    RECA_XSEM_WAN27_I2V         =  8    # docs ~9
    RECA_XSEM_WAN27_R2V         = 12
    ...

If a backend isn't listed in the config dict, it is NOT gated (falls
through to whatever per-process semaphore the backend already has).
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

_DEFAULT_BASE_DIR = Path(os.environ.get("RECA_XSEM_BASE_DIR", "/tmp/videorlm_xsem"))
_DEFAULT_TTL_S = int(os.environ.get("RECA_XSEM_TTL_S", "1800"))
_DEFAULT_POLL_S = float(os.environ.get("RECA_XSEM_POLL_S", "0.3"))


def _default_caps() -> dict[str, int]:
    """Read caps from env, fall back to safe baselines."""
    def _g(name: str, default: int) -> int:
        return int(os.environ.get(f"RECA_XSEM_{name}", str(default)))

    return {
        "happyhorse-1.0-r2v": _g("HAPPYHORSE_R2V", 18),
        "happyhorse-1.0-i2v": _g("HAPPYHORSE_I2V", 8),
        "happyhorse-1.0-t2v": _g("HAPPYHORSE_T2V", 8),
        "gpt-image-2": _g("GPT_IMAGE_2", 12),
        "gpt-image-2-pro": _g("GPT_IMAGE_2_PRO", 12),
        "wan2.7-i2v": _g("WAN27_I2V", 8),
        "wan2.7-r2v": _g("WAN27_R2V", 12),
        "wan2.7-t2v": _g("WAN27_T2V", 16),
        "wan2.7-image": _g("WAN27_IMAGE", 8),
        "wan2.7-image-pro": _g("WAN27_IMAGE_PRO", 8),
        "wan2.6-image": _g("WAN26_IMAGE", 8),
        "wanx-v1": _g("WANX_V1", 2),
        "qwen-image-2.0-pro": _g("QWEN_IMAGE_PRO", 8),
        # LLM chat agents (model name == agent_name)
        "gpt-5.5": _g("GPT_5_5", 12),
        "qwen3.6-max-preview": _g("QWEN36_MAX", 12),
        "qwen3.6-plus": _g("QWEN36_PLUS", 12),
        "qwen3.6-flash": _g("QWEN36_FLASH", 12),
    }


class _FsSemaphore:
    """Filesystem cross-process semaphore using flock + per-slot markers.

    Storage layout:
        ``<base_dir>/<backend>/lockfile``  → flock target, never deleted
        ``<base_dir>/<backend>/slot_<uuid>``  → per-acquire marker, deleted on release

    Acquire algorithm:
        1. flock(LOCK_EX) on lockfile
        2. count marker files; GC any with mtime > TTL
        3. if count < cap: create my marker, return
        4. else: unlock, sleep poll_interval, retry

    Release: delete my marker. Stale markers from crashed processes
    are auto-collected on next acquire's GC sweep (TTL=1800s default).
    """

    def __init__(
        self,
        base_dir: Path,
        caps: dict[str, int],
        ttl_s: int,
        poll_interval_s: float,
    ):
        self._base = base_dir
        self._caps = caps
        self._ttl_s = ttl_s
        self._poll_s = poll_interval_s
        self._base.mkdir(parents=True, exist_ok=True)

    def _backend_dir(self, name: str) -> Path:
        d = self._base / name.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _gc_and_count(self, backend_dir: Path) -> int:
        now = time.time()
        n = 0
        for p in backend_dir.iterdir():
            if not p.name.startswith("slot_"):
                continue
            try:
                age = now - p.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > self._ttl_s:
                try:
                    p.unlink()
                    print(f"[xsem] GC stale slot {p.name} (age {age:.0f}s)", flush=True)
                except FileNotFoundError:
                    pass
                continue
            n += 1
        return n

    @contextlib.contextmanager
    def slot(self, backend_name: str):
        cap = self._caps.get(backend_name)
        if cap is None or cap <= 0:
            yield
            return
        backend_dir = self._backend_dir(backend_name)
        lockfile = backend_dir / "lockfile"
        lockfile.touch(exist_ok=True)
        my_marker = backend_dir / f"slot_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex[:8]}"

        acquired = False
        wait_logged = False
        wait_started_at = time.time()

        try:
            while not acquired:
                with open(lockfile, "rb") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX)
                    try:
                        in_use = self._gc_and_count(backend_dir)
                        if in_use < cap:
                            my_marker.touch()
                            acquired = True
                            if wait_logged:
                                waited = time.time() - wait_started_at
                                print(
                                    f"[xsem] {backend_name} acquired after {waited:.1f}s wait "
                                    f"({in_use + 1}/{cap})",
                                    flush=True,
                                )
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
                if acquired:
                    break
                if not wait_logged:
                    print(
                        f"[xsem] {backend_name} pool full ({cap}), waiting...",
                        flush=True,
                    )
                    wait_logged = True
                time.sleep(self._poll_s)
            yield
        finally:
            try:
                my_marker.unlink()
            except FileNotFoundError:
                pass


class _NoopSemaphore:
    @contextlib.contextmanager
    def slot(self, backend_name: str):
        yield


_INSTANCE: object = _NoopSemaphore()
_PATCHED = False


def slot(backend_name: str):
    """Acquire a cross-process slot for `backend_name`. Use as a context manager.

    In single-process / unconfigured mode, this is a noop yield. After
    ``enable_xsem()`` is called, it acquires a flock-backed slot.
    """
    return _INSTANCE.slot(backend_name)


def enable_xsem(
    base_dir: str | Path | None = None,
    caps: dict[str, int] | None = None,
    ttl_s: int | None = None,
    poll_interval_s: float | None = None,
) -> None:
    """Activate cross-process semaphore + monkey-patch dispatch / QwenAgent.

    Idempotent — calling twice is a noop.

    Important: must be called BEFORE any module imports
    ``from videorlm.backends.media import dispatch`` etc., otherwise
    the import will bind to the original (un-wrapped) function. The
    helpers in ``_smoke.py`` call this at process start before
    importing ``videorlm.framework``.
    """
    global _INSTANCE, _PATCHED
    if _PATCHED:
        return

    base = Path(base_dir or _DEFAULT_BASE_DIR)
    cap_dict = caps if caps is not None else _default_caps()
    _INSTANCE = _FsSemaphore(
        base_dir=base,
        caps=cap_dict,
        ttl_s=ttl_s if ttl_s is not None else _DEFAULT_TTL_S,
        poll_interval_s=poll_interval_s if poll_interval_s is not None else _DEFAULT_POLL_S,
    )
    print(
        f"[xsem] enabled, base_dir={base}, "
        f"{len([k for k, v in cap_dict.items() if v > 0])} backends gated",
        flush=True,
    )
    _patch_media_dispatch()
    _patch_agents()
    _PATCHED = True


def _patch_media_dispatch() -> None:
    """Wrap dispatch_segment / dispatch_bridge / dispatch_image with xsem."""
    # NOTE: we must import the submodule, not the re-exported function with
    # the same name in interface/__init__.py.
    import importlib
    dispatch_mod = importlib.import_module("videorlm.backends.media.interface.dispatch")

    def _resolve_backend_for_request(request):
        """Pick the env-routed backend for a request. Mirrors dispatch.py
        env mapping: SegmentRequest → RECA_RENDER_BACKEND_SEGMENT_{MODE},
        BridgeRequest → ..._BRIDGE, ImageRequest → ..._{KIND}."""
        cls_name = type(request).__name__
        mode = getattr(request, "mode", None)
        kind = getattr(request, "kind", None)
        try:
            if cls_name == "SegmentRequest":
                env_key = f"RECA_RENDER_BACKEND_SEGMENT_{(mode or 'r2v').upper()}"
                return os.environ.get(env_key)
            if cls_name == "BridgeRequest":
                return os.environ.get("RECA_RENDER_BACKEND_BRIDGE")
            if cls_name == "ImageRequest":
                return os.environ.get(f"RECA_RENDER_BACKEND_{(kind or '').upper()}")
        except Exception:
            return None
        return None

    def _wrap(orig_fn: Callable):
        def wrapped(request, *args, **kwargs):
            resolved = _resolve_backend_for_request(request)
            if resolved is None:
                return orig_fn(request, *args, **kwargs)
            with slot(resolved):
                return orig_fn(request, *args, **kwargs)
        wrapped.__name__ = getattr(orig_fn, "__name__", "wrapped")
        wrapped.__doc__ = getattr(orig_fn, "__doc__", "")
        return wrapped

    wrapped_segment = _wrap(dispatch_mod.dispatch_segment)
    wrapped_bridge = _wrap(dispatch_mod.dispatch_bridge)
    wrapped_image = _wrap(dispatch_mod.dispatch_image)

    dispatch_mod.dispatch_segment = wrapped_segment
    dispatch_mod.dispatch_bridge = wrapped_bridge
    dispatch_mod.dispatch_image = wrapped_image

    # also propagate to package-level re-exports if already imported
    try:
        import videorlm.backends.media as media_pkg
        media_pkg.dispatch_segment = wrapped_segment
        media_pkg.dispatch_bridge = wrapped_bridge
        media_pkg.dispatch_image = wrapped_image
    except Exception:
        pass

    print("[xsem] media dispatch wrapped (3 fns: segment/bridge/image)", flush=True)


def _patch_agents() -> None:
    """Wrap ``OpenAICompatibleAgent.prompt`` with xsem (gated by config.model name).

    Patches the base class once; both ``OpenAICompatibleAgent`` and the
    DashScope subclass (``DashScopeQwenAgent``) inherit the wrapped method
    via MRO, so a single wrap covers every OpenAI-compatible flavor.
    """
    try:
        from videorlm.backends.llm.agents.openai_compat.agent import OpenAICompatibleAgent
    except Exception:
        return

    if getattr(OpenAICompatibleAgent.prompt, "_xsem_wrapped", False):
        return

    orig_prompt = OpenAICompatibleAgent.prompt

    def wrapped_prompt(self, text: str) -> str:
        backend_name = getattr(self._config, "model", None)
        if not backend_name:
            return orig_prompt(self, text)
        with slot(backend_name):
            return orig_prompt(self, text)

    wrapped_prompt._xsem_wrapped = True  # type: ignore[attr-defined]
    OpenAICompatibleAgent.prompt = wrapped_prompt
    print(
        "[xsem] OpenAICompatibleAgent.prompt wrapped "
        "(covers DashScopeQwenAgent via MRO)",
        flush=True,
    )


def maybe_enable_from_env() -> None:
    """If ``RECA_XSEM_ENABLE`` env var is truthy, call enable_xsem() now.

    Designed to be called at the very top of process entry points
    (e.g. _smoke.py main()) so the patches land BEFORE pipeline.py
    imports happen further down the call chain.
    """
    if os.environ.get("RECA_XSEM_ENABLE", "").lower() in ("1", "true", "yes"):
        enable_xsem()


__all__ = ["enable_xsem", "maybe_enable_from_env", "slot"]
