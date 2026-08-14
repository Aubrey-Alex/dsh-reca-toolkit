"""Shared helpers for LLM and video backend packages."""

from .errors import BackendError, StructuralBackendError, TransientBackendError
from .key_pool import KeyPool, KeyState, cached_key_pool, classify_error
from .platforms import PlatformProfile, get_platform, with_key
from .registry import BackendRegistry
from .retry import RetryPolicy, retry_until_exhausted
from .timeout import (
    BackendCallTimeout,
    call_with_timeout,
    default_timeout_s,
)

__all__ = [
    "BackendError",
    "BackendRegistry",
    "BackendCallTimeout",
    "KeyPool",
    "KeyState",
    "PlatformProfile",
    "RetryPolicy",
    "StructuralBackendError",
    "TransientBackendError",
    "call_with_timeout",
    "cached_key_pool",
    "classify_error",
    "default_timeout_s",
    "get_platform",
    "retry_until_exhausted",
    "with_key",
]
