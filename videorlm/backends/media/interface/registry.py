"""Backend registry: name → backend instance + kind → backend resolution.

Provider implementations under ``backends/media/impl/*`` register themselves
at import time. Resolution for ``for_kind(kind)`` has exactly **two** layers:

  1. env  ``RECA_RENDER_BACKEND_<KIND>=name``  (the one caller-facing knob)
  2. ``DEFAULT_RENDER_BACKEND`` (this module's hardcoded default)

No JSON config file, no per-directory overrides, no env→config→default
cascade. One env var per kind, or fall through to the default. Anything
beyond that was making backend selection un-auditable.

Stored backend instances can be any class — segment backends conform to
``VideoSegmentBackend``; image backends are plain classes with a
``render(ImageRequest)`` method.
"""
from __future__ import annotations

import os
from typing import Any

from ..._common.registry import BackendRegistry
from .capabilities import RenderPlanError


# Default kind → backend routing table. Self-contained so backends/ stays
# independent. Override per-kind via env RECA_RENDER_BACKEND_<KIND> only.
#
# Kinds in this table correspond to dispatch entrypoints:
#   - segment_i2v / segment_r2v  ⇒ dispatch_segment(req with mode=...)
#   - bridge                     ⇒ dispatch_bridge(req)
#   - portrait / anchor_image / location / prop / image_edit
#                                ⇒ dispatch_image(req)
DEFAULT_RENDER_BACKEND: dict[str, str] = {
    "anchor_image":  "wan2.7-image",
    "portrait":      "wan2.7-image",
    "location":      "wan2.7-image",
    "prop":          "wan2.7-image",
    "image_edit":    "wan2.7-image",
    "segment_i2v":   "happyhorse-1.0-i2v",   # serial_segment: first-only chain
    "segment_r2v":   "wan2.7-r2v",            # reference_serial_segment: first + refs
    "bridge":        "wan2.7-i2v",
}


_REGISTRY = BackendRegistry()


def register_backend(name: str, backend: Any) -> None:
    """Register a backend by canonical name (e.g. "wan2.7-i2v")."""
    _REGISTRY.register(backend, name=name)


def get_backend(name: str) -> Any:
    """Lookup by exact name. Lazy-loads the impl/ tree on first miss."""
    backend = _REGISTRY.get(name)
    if backend is None:
        _autoload_default_backends()
        backend = _REGISTRY.get(name)
    if backend is None:
        raise KeyError(
            f"Backend {name!r} not registered. "
            f"Available: {_REGISTRY.list_names()}"
        )
    return backend  # type: ignore[return-value]


def list_backends() -> list[str]:
    if not _REGISTRY.list_names():
        _autoload_default_backends()
    return _REGISTRY.list_names()


def for_kind(kind: str) -> Any:
    """Pick a registered backend for a RenderKind.

    Resolution: ``RECA_RENDER_BACKEND_<KIND>`` env > ``DEFAULT_RENDER_BACKEND``.
    """
    env_key = f"RECA_RENDER_BACKEND_{kind.upper()}"
    name = os.environ.get(env_key) or DEFAULT_RENDER_BACKEND.get(kind)
    if not name:
        raise RenderPlanError(
            f"No default backend mapping for RenderKind {kind!r}"
        )
    backend = _REGISTRY.get(name)
    if backend is None:
        _autoload_default_backends()
        backend = _REGISTRY.get(name)
    if backend is None:
        raise RenderPlanError(
            f"Backend {name!r} (chosen for kind {kind!r}) is not "
            f"registered. Available: {_REGISTRY.list_names()}"
        )
    return backend  # type: ignore[return-value]


def _autoload_default_backends() -> None:
    """Import the impl/ tree to trigger backend registrations.

    Cheap: provider modules just construct instances and call
    ``register_backend`` — no network calls.
    """
    try:
        from .. import impl  # noqa: F401
        from ..impl import dashscope  # noqa: F401
        try:
            from ..impl import openai as _openai_media  # noqa: F401
        except Exception as e:  # noqa: BLE001
            # openai-compat media backends are optional (depends on openai
            # sdk + an OpenAI_API_KEY-class env). Don't fail the whole
            # autoload if it can't be loaded.
            print(f"[backends.media/registry] openai autoload skipped: {e}",
                  flush=True)
        try:
            from ..impl import local  # noqa: F401
        except Exception as e:  # noqa: BLE001
            # local backends are placeholders until weights / runtime land.
            print(f"[backends.media/registry] local autoload skipped: {e}",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[backends.media/registry] autoload failed: {e}", flush=True)
