"""Media backend public API.

Pure abstractions live in ``interface/``; concrete backends live in
``impl/<vendor>/<...>/<model>.py``. This top-level module re-exports the
interface symbols expected by the framework agent.

Public surface (locked names — do NOT rename / alias):

    SegmentRequest, BridgeRequest, ImageRequest
    ImageRef, VideoResult, ImageResult, ProviderSpec
    VideoSegmentBackend, VideoSegmentBackendBase
    dispatch_segment, dispatch_bridge, dispatch_image
    get_backend, list_backends, for_kind, register_backend
"""
from __future__ import annotations

from .interface import (
    BackendCapabilities,
    BackendRenderError,
    BridgeRequest,
    ImageKind,
    ImageRef,
    ImageRefRole,
    ImageRequest,
    ImageResult,
    ProviderSpec,
    RenderPlanError,
    SegmentMode,
    SegmentRequest,
    VideoResult,
    VideoSegmentBackend,
    VideoSegmentBackendBase,
    dispatch_bridge,
    dispatch_image,
    dispatch_segment,
    for_kind,
    get_backend,
    list_backends,
    register_backend,
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
