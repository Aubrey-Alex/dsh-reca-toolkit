"""Provider registry — short provider name → ``Provider`` instance.

Built-in providers self-register at import time. New providers (in-tree
or downstream) call ``register_provider(Provider(...))`` at module load.

Lookup is case-insensitive.

Example::

    from videorlm.backends._common.providers import get_provider
    pool = get_provider("dashscope").key_pool()
    limits = get_provider("openai").rate_limits()
"""
from __future__ import annotations

import threading

from ._base import Provider


_REGISTRY: dict[str, Provider] = {}
_LOCK = threading.Lock()


def register_provider(provider: Provider, *, replace: bool = False) -> None:
    key = provider.name.strip().lower()
    if not key:
        raise ValueError("Provider.name must be non-empty")
    with _LOCK:
        if key in _REGISTRY and not replace:
            raise ValueError(
                f"provider {key!r} already registered; pass replace=True to override"
            )
        _REGISTRY[key] = provider


def get_provider(name: str) -> Provider:
    key = (name or "").strip().lower()
    with _LOCK:
        if key not in _REGISTRY:
            available = sorted(_REGISTRY.keys())
            raise KeyError(
                f"unknown provider {name!r}; available={available}"
            )
        return _REGISTRY[key]


def list_providers() -> list[str]:
    with _LOCK:
        return sorted(_REGISTRY.keys())


# Auto-register the built-in providers. Each module calls
# ``register_provider(...)`` at import time as a side-effect; we use
# ``importlib.import_module`` so pyflakes doesn't flag the imports as
# unused (it doesn't understand the side-effect-only pattern).
import importlib

for _name in ("dashscope", "openai", "_passthrough"):
    importlib.import_module(f"{__name__}.{_name}")
del _name, importlib


__all__ = [
    "Provider",
    "get_provider",
    "list_providers",
    "register_provider",
]
