"""Public lifecycle schemas shared by the Gateway and the DSH plugin."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


GATEWAY_STATES = {
    "queued",
    "running",
    "interrupted",
    "cancelling",
    "cancelled",
    "failed",
    "succeeded",
}

RECA_STAGES = {
    "planning",
    "asset_generation",
    "rendering",
    "validating",
    "repairing",
    "concat",
    "succeeded",
    "failed",
}

AUDIT_STATES = {
    "audit_pending",
    "audit_running",
    "audit_retrying",
    "audit_failed",
    "audit_skipped",
    "audit_repaired",
    "audited",
}


@dataclass(frozen=True)
class RunConfig:
    """User-level request config passed into ReCA without exposing secrets."""

    duration_s: int | None = None
    resolution: str = "1280x720"
    style: str = "cinematic"
    aspect_ratio: str = "16:9"
    backend: str = "wan"
    seed: int = 0
    enable_audit: bool = True
    validate_segments: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_run_config(raw: dict[str, Any] | None) -> RunConfig:
    raw = raw if isinstance(raw, dict) else {}
    duration = raw.get("duration_s", raw.get("duration"))
    if duration in (None, ""):
        duration_s = None
    else:
        duration_s = int(duration)
        if duration_s < 3 or duration_s > 3600:
            raise ValueError("duration must be between 3 and 3600 seconds")
    resolution = str(raw.get("resolution") or "1280x720").strip()
    style = str(raw.get("style") or "cinematic").strip()
    aspect_ratio = str(raw.get("aspect_ratio") or "16:9").strip()
    backend = str(raw.get("backend") or "wan").strip()
    if backend not in {"wan", "wan27", "happyhorse", "ltx"}:
        raise ValueError("backend must be wan, wan27, happyhorse, or ltx")
    return RunConfig(
        duration_s=duration_s,
        resolution=resolution,
        style=style,
        aspect_ratio=aspect_ratio,
        backend=backend,
        seed=int(raw.get("seed", 0) or 0),
        enable_audit=bool(raw.get("enable_audit", raw.get("validate", True))),
        validate_segments=bool(raw.get("validate_segments", False)),
    )
