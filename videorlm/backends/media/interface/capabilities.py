"""BackendCapabilities + structural / runtime backend errors.

`dispatch` reads BackendCapabilities at planning time to decide whether a
request is dispatchable. Capability mismatches raise `RenderPlanError`
*before* any model call so we don't waste a slot.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._common.errors import BackendError, StructuralBackendError


# ── BackendCapabilities ──────────────────────────────────────────────


@dataclass(frozen=True)
class BackendCapabilities:
    """What a concrete VideoBackend can do."""

    backend_name: str
    model_family: str
    supports_kinds: frozenset[str]
    provider: str = ""
    model_id: str = ""

    # Modality flags
    supports_t2v: bool = False
    supports_i2v: bool = False
    supports_r2v: bool = False
    supports_first_image: bool = False
    supports_last_image: bool = False
    supports_mid_reference: bool = False
    supports_multi_shot_single_call: bool = False
    supports_t2i: bool = False
    supports_i2i: bool = False

    # Anchor semantics tag for observability only. Not consumed by dispatch.
    flf2v_mode: str = "none"

    # Duration constraints
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0
    duration_granularity_s: float = 1.0

    # Format / size
    supported_resolutions: tuple[str, ...] = ()
    max_prompt_chars: int = 0
    max_reference_images: int = 0

    # Throughput
    max_concurrency: int = 1
    requests_per_minute: int = 0

    # Cost
    estimated_cost_per_call_usd: float = 0.0
    estimated_cost_per_second_usd: float = 0.0

    # Retry policy used by dispatch wrapper.
    max_retries: int = 3
    retry_backoff_base_s: float = 1.5
    retry_backoff_max_s: float = 60.0
    retryable_error_names: tuple[str, ...] = (
        "BackendRenderError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "TimeoutError",
        "HTTPError",
        "TooManyRedirects",
    )


# ── Errors ────────────────────────────────────────────────────────────


class RenderPlanError(StructuralBackendError, ValueError):
    """Planning-time mismatch between request and backend capabilities.
    Raised before any model call so we don't waste a slot."""


class BackendRenderError(BackendError, RuntimeError):
    """The backend's underlying API returned an error (rate limit, content
    filter, network, etc.). Distinct from RenderPlanError so RepairRouter
    can dispatch differently."""
