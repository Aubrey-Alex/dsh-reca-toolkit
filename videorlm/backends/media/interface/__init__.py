"""Video backend interface layer (pure abstractions, zero SDK deps).

Public API surface for the rest of the codebase. Concrete backends live
in ``backends/media/impl/<vendor>/`` and register themselves via
``register_backend``.
"""
from __future__ import annotations

from .capabilities import (
    BackendCapabilities,
    BackendRenderError,
    RenderPlanError,
)
from .requests import (
    BridgeRequest,
    ImageKind,
    ImageRef,
    ImageRefRole,
    ImageRequest,
    ImageResult,
    SegmentMode,
    SegmentRequest,
    VideoResult,
)
from .registry import (
    for_kind,
    get_backend,
    list_backends,
    register_backend,
)
from .segment_backend import (
    ProviderSpec,
    VideoSegmentBackend,
    VideoSegmentBackendBase,
)
from .dispatch import (
    dispatch_bridge,
    dispatch_image,
    dispatch_segment,
)


__all__ = [
    # types
    "BackendCapabilities",
    "BridgeRequest",
    "ImageKind",
    "ImageRef",
    "ImageRefRole",
    "ImageRequest",
    "ImageResult",
    "ProviderSpec",
    "SegmentMode",
    "SegmentRequest",
    "VideoResult",
    "VideoSegmentBackend",
    "VideoSegmentBackendBase",
    # errors
    "RenderPlanError",
    "BackendRenderError",
    # registry
    "register_backend",
    "get_backend",
    "list_backends",
    "for_kind",
    # dispatch
    "dispatch_segment",
    "dispatch_bridge",
    "dispatch_image",
]
