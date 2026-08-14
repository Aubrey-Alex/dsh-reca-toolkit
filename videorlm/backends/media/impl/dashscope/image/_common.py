"""Image-backend common helpers."""
from __future__ import annotations

from ....interface.requests import ImageRequest


NO_REF_HINT = (
    "no reference image provided; this backend requires ≥1 ref. "
    "For pure text-to-image use a T2I-capable backend (wan2.7-image / "
    "gpt-image-2 / qwen-image-2.0)."
)


def request_size(request: ImageRequest, default: str = "1024*1024") -> str:
    if not request.resolution:
        return default
    return str(request.resolution).replace("x", "*")


def collect_image_refs(request: ImageRequest) -> list[str]:
    """Pull URLs out of an ImageRequest.references tuple in order.

    Roles eligible: portrait / scene / reference / source. Any other role
    is silently ignored.
    """
    allowed = {"source", "portrait", "scene", "reference"}
    return [r.url for r in request.references if r.role in allowed]
