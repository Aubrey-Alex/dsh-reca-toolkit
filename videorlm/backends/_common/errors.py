"""Shared backend error taxonomy."""


class BackendError(Exception):
    """Base class for backend-layer errors."""


class TransientBackendError(BackendError):
    """Retry-eligible backend error: rate limit, timeout, temporary outage."""


class StructuralBackendError(BackendError):
    """Fail-fast backend error: invalid request or unsupported capability."""
