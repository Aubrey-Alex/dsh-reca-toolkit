"""Render dispatch — four typed entrypoints, no chain, no fallback.

Public surface:

  - ``dispatch_segment(SegmentRequest) -> VideoResult``
      Routes by ``RECA_RENDER_BACKEND_SEGMENT_<MODE>`` (i2v / r2v) when
      defined, else falls through to ``RECA_RENDER_BACKEND_SEGMENT``. Calls
      ``backend.render_segment(req)``.

  - ``dispatch_bridge(BridgeRequest) -> VideoResult``
      Routes by ``RECA_RENDER_BACKEND_BRIDGE``. Calls
      ``backend.render_bridge(req)``.

  - ``dispatch_image(ImageRequest) -> ImageResult``
      Routes by ``RECA_RENDER_BACKEND_<KIND>``. Calls
      ``backend.render(req)``.

All three go through ``retry_until_exhausted`` on the SAME backend. No
cross-backend fallback. No chain. If the selected backend's transient
retries are exhausted, ``RuntimeError`` is raised; switching backends is
the framework's responsibility (and must happen out-of-band, not inside
this module).
"""
from __future__ import annotations

import os
import threading
from typing import Callable

from ..._common.retry import RetryPolicy, retry_until_exhausted
from ..._common.timeout import call_with_timeout, default_timeout_s
from ..._common.trace import trace_event
from .capabilities import BackendCapabilities, RenderPlanError
from .registry import for_kind, get_backend
from .requests import (
    BridgeRequest,
    ImageRequest,
    ImageResult,
    SegmentRequest,
    VideoResult,
)


# ── Per-backend semaphore (sized by caps.max_concurrency) ─────────────────
#
# Backed by ``NamedSemaphorePool`` (``_common/concurrency.py``). Env override
# ``RECA_BACKEND_CONCURRENCY_<NAME>`` trumps the caller's ``caps.max_concurrency``;
# NAME is uppercased with ``-``/``.`` → ``_``.

from ..._common.concurrency import NamedSemaphorePool


_BACKEND_SLOTS = NamedSemaphorePool(env_prefix="RECA_BACKEND_CONCURRENCY_")


def _get_semaphore(
    backend_name: str, caps: BackendCapabilities,
) -> threading.BoundedSemaphore:
    name = backend_name or caps.backend_name or "_unnamed"
    return _BACKEND_SLOTS.semaphore_for(name, caps.max_concurrency)


# ── Retry policy for video dispatch ───────────────────────────────────────


_TRANSIENT_MARKERS = frozenset({
    "ratelimit", "rate limit", "throttling", "throttled",
    "datainspectionfailed", "content filter", "concurrentlimit",
    "timeout", "temporarily unavailable", "503", "502", "429",
    "ratequota",
})


def _make_policy(caps: BackendCapabilities) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max(1, int(caps.max_retries) + 1),
        backoff_base_s=float(caps.retry_backoff_base_s),
        backoff_cap_s=float(caps.retry_backoff_max_s),
        transient_msg_substrings=_TRANSIENT_MARKERS.union(
            marker.lower() for marker in caps.retryable_error_names
        ),
    )


def _video_timeout_s() -> float:
    # Outer per-attempt timeout for dispatch_*. Must be >= the longest inner
    # backend poll budget; happyhorse-1.0-r2v polls up to 3600s, so default
    # must clear that with some headroom for submit + download.
    return float(default_timeout_s("RECA_VIDEO_CALL_TIMEOUT_S", 3700.0))


def _autofill_cost(result: VideoResult, caps: BackendCapabilities,
                   request_duration: float) -> None:
    if result.cost_usd != 0.0:
        return
    est_call = float(caps.estimated_cost_per_call_usd or 0.0)
    est_per_s = float(caps.estimated_cost_per_second_usd or 0.0)
    duration_for_cost = float(result.rendered_duration_s or request_duration or 0.0)
    result.cost_usd = est_call + est_per_s * duration_for_cost


# ── Shared wrapper (no chain — single backend, retry only) ────────────────


def _run_single_backend(
    *,
    op_name: str,
    backend_name: str,
    caps: BackendCapabilities,
    request_id: str,
    log_dir: str | None,
    duration_s: float,
    render_call: Callable[[], VideoResult | ImageResult],
    label_suffix: str,
    start_extras: dict | None = None,
):
    sem = _get_semaphore(backend_name, caps)
    policy = _make_policy(caps)
    timeout_s = _video_timeout_s()

    trace_event(
        op_name, "start", log_dir=log_dir,
        request_id=request_id,
        backend=backend_name, provider=caps.provider,
        timeout_s=timeout_s, duration_s=duration_s,
        **(start_extras or {}),
    )

    def _attempt():
        trace_event(
            op_name, "attempt_start", log_dir=log_dir,
            request_id=request_id, backend=backend_name,
        )
        return call_with_timeout(
            render_call,
            timeout_s=timeout_s,
            label=f"{op_name}[{backend_name}:{label_suffix}]",
        )

    try:
        with sem:
            result = retry_until_exhausted(backend_name, _attempt, policy=policy)
        if hasattr(result, "backend_name"):
            result.backend_name = result.backend_name or backend_name
        if isinstance(result, VideoResult):
            _autofill_cost(result, caps, duration_s)
        trace_event(
            op_name, "success", log_dir=log_dir,
            request_id=request_id, backend=backend_name,
            output_path=getattr(result, "output_path", "") or "",
            output_url=bool(getattr(result, "output_url", None)),
            cost_usd=getattr(result, "cost_usd", 0.0),
        )
        return result
    except Exception as exc:
        trace_event(
            op_name, "error", log_dir=log_dir,
            request_id=request_id, backend=backend_name,
            error_type=type(exc).__name__, error=str(exc)[:500],
        )
        raise


# ── Backend resolution helpers ────────────────────────────────────────────


def _resolve_segment_backend(mode: str):
    """Resolve segment backend from env.

    Priority:
      1. ``RECA_RENDER_BACKEND_SEGMENT_<MODE>`` (e.g. ..._I2V / ..._R2V)
      2. ``RECA_RENDER_BACKEND_SEGMENT``
      3. ``for_kind("segment_<mode>")`` (legacy default routing table)
    """
    env_split = os.environ.get(f"RECA_RENDER_BACKEND_SEGMENT_{mode.upper()}", "").strip()
    if env_split:
        return env_split, get_backend(env_split)
    env_plain = os.environ.get("RECA_RENDER_BACKEND_SEGMENT", "").strip()
    if env_plain:
        return env_plain, get_backend(env_plain)
    # Fallback to the kind-mapped registry (DEFAULT_RENDER_BACKEND).
    backend = for_kind(f"segment_{mode}")
    name = getattr(backend, "name", None) or _backend_name_for(backend)
    return name, backend


def _resolve_named_backend(env_key: str, kind: str | None = None):
    name = os.environ.get(env_key, "").strip()
    if name:
        return name, get_backend(name)
    if kind:
        backend = for_kind(kind)
        return _backend_name_for(backend), backend
    raise RenderPlanError(f"env {env_key!r} is not set and no kind fallback given")


def _backend_name_for(backend) -> str:
    """Pull a stable name out of any backend object (Protocol or legacy)."""
    spec = getattr(backend, "PROVIDER_SPEC", None)
    if spec is not None:
        name = getattr(spec, "name", None)
        if name:
            return str(name)
    name = getattr(backend, "name", None)
    if isinstance(name, str) and name:
        return name
    name = getattr(backend, "NAME", None)
    if isinstance(name, str) and name:
        return name
    try:
        caps = backend.capabilities()
        if caps.backend_name:
            return caps.backend_name
    except Exception:
        pass
    return type(backend).__name__


def _caps_for(backend) -> BackendCapabilities:
    """Read BackendCapabilities — works for new Protocol + image backends."""
    return backend.capabilities()


# ── Public dispatch entrypoints ───────────────────────────────────────────


def dispatch_segment(request: SegmentRequest) -> VideoResult:
    """Render a single segment (i2v or r2v). Single backend, retry-only."""
    backend_name, backend = _resolve_segment_backend(request.mode)
    if not hasattr(backend, "render_segment"):
        raise RenderPlanError(
            f"backend={backend_name} does not implement render_segment "
            f"(required for SegmentRequest)"
        )
    caps = _caps_for(backend)
    return _run_single_backend(
        op_name="video.dispatch_segment",
        backend_name=backend_name,
        caps=caps,
        request_id=request.request_id,
        log_dir=request.log_dir,
        duration_s=request.duration_s,
        render_call=lambda: backend.render_segment(request),
        label_suffix=f"segment:{request.mode}:{request.request_id}",
        start_extras={
            "mode": request.mode,
            "output_path": request.output_path or "",
            "n_refs": len(request.reference_image_urls),
        },
    )


def dispatch_bridge(request: BridgeRequest) -> VideoResult:
    """Render a chunk-boundary bridge (first + last). Single backend, retry-only."""
    backend_name, backend = _resolve_named_backend(
        "RECA_RENDER_BACKEND_BRIDGE", kind="bridge",
    )
    if not hasattr(backend, "render_bridge"):
        raise RenderPlanError(
            f"backend={backend_name} does not implement render_bridge "
            f"(required for BridgeRequest)"
        )
    caps = _caps_for(backend)
    return _run_single_backend(
        op_name="video.dispatch_bridge",
        backend_name=backend_name,
        caps=caps,
        request_id=request.request_id,
        log_dir=request.log_dir,
        duration_s=request.duration_s,
        render_call=lambda: backend.render_bridge(request),
        label_suffix=f"bridge:{request.request_id}",
        start_extras={
            "output_path": request.output_path or "",
        },
    )


def dispatch_image(request: ImageRequest) -> ImageResult:
    """Render an image (portrait / anchor / etc). Single backend, retry-only."""
    env_key = f"RECA_RENDER_BACKEND_{request.kind.upper()}"
    backend_name, backend = _resolve_named_backend(env_key, kind=request.kind)
    if not hasattr(backend, "render"):
        raise RenderPlanError(
            f"backend={backend_name} does not implement render "
            f"(required for ImageRequest)"
        )
    caps = _caps_for(backend)
    return _run_single_backend(
        op_name="video.dispatch_image",
        backend_name=backend_name,
        caps=caps,
        request_id=request.request_id,
        log_dir=request.log_dir,
        duration_s=0.0,
        render_call=lambda: backend.render(request),
        label_suffix=f"image:{request.kind}:{request.request_id}",
        start_extras={
            "kind": request.kind,
            "output_path": request.output_path or "",
            "refs": len(request.references),
        },
    )


__all__ = [
    "dispatch_segment",
    "dispatch_bridge",
    "dispatch_image",
]
