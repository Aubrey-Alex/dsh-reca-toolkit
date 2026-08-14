"""VideoSegmentBackend — the only Protocol every video render backend must satisfy.

Drop-in template for video providers. Backends ONLY implement these 2 render
methods + 1 spec block:

  - ``render_segment(SegmentRequest)``   — first frame + optional refs → mp4
  - ``render_bridge(BridgeRequest)``     — first + last frame → mp4 transition

All other concerns (image_edit / storyboard / t2v_baseline) live
elsewhere (image backends or backend-side extensions).

See ``docs/backends.md §5``.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .capabilities import BackendCapabilities
from .requests import BridgeRequest, SegmentRequest, VideoResult


# ── ProviderSpec ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProviderSpec:
    """Static capability spec for a video segment backend.

    Used to (a) declare what the backend can render, (b) auto-derive a
    ``BackendCapabilities`` blob for the legacy interface (dispatch
    plan-time checks) without forcing the backend author to construct one.
    """

    name: str                               # registered name (RECA_RENDER_BACKEND_* value)
    family: str
    supports_resolutions: tuple[str, ...] = ()
    segment_duration_range: tuple[float, float] = (3.0, 15.0)
    bridge_duration_range: tuple[float, float] = (3.0, 8.0)
    max_reference_images: int = 0
    max_prompt_chars: int = 2000
    concurrency_env: str = ""
    default_concurrency: int = 8
    # When > 0, the backend sem cap is computed as
    # ``per_key_concurrency × N_active_keys`` for ``provider`` (auto-scales
    # when the operator adds/removes API keys). Used by DashScope video
    # backends (happyhorse, wan) where the realistic ceiling is "8 concurrent
    # tasks per key". Falls back to ``default_concurrency`` if the provider
    # has no registered keys yet, and is overridden by ``concurrency_env``
    # if that env is set (the explicit override always wins).
    per_key_concurrency: int = 0
    cost_per_call_usd: float = 0.0
    cost_per_second_usd: float = 0.0
    provider: str = ""
    notes: str = ""
    # i2v / r2v capability bitmap (drives mode dispatching inside backend)
    supports_i2v: bool = True
    supports_r2v: bool = True

    def resolved_concurrency(self) -> int:
        # 1. Explicit env override (highest priority — old behaviour).
        if self.concurrency_env:
            raw = os.environ.get(self.concurrency_env, "").strip()
            if raw:
                try:
                    return max(1, int(raw))
                except ValueError:
                    pass
        # 2. Auto-scale by number of active API keys for this provider.
        if self.per_key_concurrency > 0 and self.provider:
            n_keys = _provider_key_count(self.provider)
            if n_keys > 0:
                return max(1, int(self.per_key_concurrency) * n_keys)
        # 3. Static fallback.
        return max(1, int(self.default_concurrency))


def _provider_key_count(provider: str) -> int:
    """How many usable API keys are currently registered for ``provider``.

    Used by ``ProviderSpec.resolved_concurrency`` to scale the backend
    semaphore as ``per_key_concurrency × N_keys``. Returns 0 if the
    provider isn't registered or its key pool isn't bound yet — callers
    treat that as "fall back to ``default_concurrency``".
    """
    try:
        from ..._common.platforms import get_provider
        pool = get_provider(provider).key_pool()
    except Exception:  # noqa: BLE001
        return 0
    return len(getattr(pool, "keys", ()) or ())


# ── Protocol (structural) ────────────────────────────────────────────────


@runtime_checkable
class VideoSegmentBackend(Protocol):
    """Structural protocol: anything with these two render methods + spec
    is a video segment backend. Concrete backends should also subclass
    ``VideoSegmentBackendBase`` for the boilerplate.
    """

    PROVIDER_SPEC: ProviderSpec

    def render_segment(self, req: SegmentRequest) -> VideoResult: ...

    def render_bridge(self, req: BridgeRequest) -> VideoResult: ...


# ── Base class with default capability adapter ────────────────────────────


class VideoSegmentBackendBase(ABC):
    """Concrete-base for video segment backends.

    Inheriting from this is recommended (vs satisfying the bare Protocol)
    because the base auto-derives the legacy ``BackendCapabilities`` blob
    from ``PROVIDER_SPEC``, sparing each backend author from writing the
    same boilerplate twice.
    """

    PROVIDER_SPEC: ProviderSpec  # subclasses must set this

    @property
    def name(self) -> str:
        return self.PROVIDER_SPEC.name

    def capabilities(self) -> BackendCapabilities:
        spec = self.PROVIDER_SPEC
        # The legacy BackendCapabilities object is still consumed by some
        # observability code (registry / xsem); produce a faithful translation.
        seg_min, seg_max = spec.segment_duration_range
        return BackendCapabilities(
            backend_name=spec.name,
            model_family=spec.family,
            supports_kinds=frozenset({"segment", "bridge"}),
            provider=spec.provider or "",
            model_id=spec.name,
            supports_t2v=False,
            supports_i2v=spec.supports_i2v,
            supports_r2v=spec.supports_r2v,
            supports_first_image=True,
            supports_last_image=False,
            supports_mid_reference=spec.supports_r2v,
            min_duration_s=float(seg_min),
            max_duration_s=float(seg_max),
            duration_granularity_s=1.0,
            supported_resolutions=spec.supports_resolutions,
            max_prompt_chars=spec.max_prompt_chars,
            max_reference_images=spec.max_reference_images,
            max_concurrency=spec.resolved_concurrency(),
            requests_per_minute=0,
            estimated_cost_per_call_usd=spec.cost_per_call_usd,
            estimated_cost_per_second_usd=spec.cost_per_second_usd,
        )

    @abstractmethod
    def render_segment(self, req: SegmentRequest) -> VideoResult:
        """First frame + optional refs (mode = i2v | r2v) → mp4."""

    @abstractmethod
    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        """First + last frame → transition mp4."""


# ── Auto-register helper ──────────────────────────────────────────────────


def auto_register(backend: VideoSegmentBackend) -> None:
    """Register the backend under its ProviderSpec name. Idempotent."""
    from .registry import register_backend
    register_backend(backend.PROVIDER_SPEC.name, backend)


__all__ = [
    "ProviderSpec",
    "VideoSegmentBackend",
    "VideoSegmentBackendBase",
    "auto_register",
]


_ = field  # keep dataclasses import non-redundant
