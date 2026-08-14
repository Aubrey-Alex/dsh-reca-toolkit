"""Backend-agnostic request / response dataclasses.

Four request shapes correspond to the four dispatch entry points:

  - ``SegmentRequest`` — first frame + optional refs → mp4
                          (i2v or r2v, decided by ``mode``)
  - ``BridgeRequest``  — first + last frame → transition mp4
  - ``ImageRequest``   — portrait / anchor_image / location / prop / image_edit

There is no other request shape. Storyboard / t2v_baseline / first-last
routing hints are deleted. Legacy request classes are gone — caller-side
rewrites are mandatory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Image references (shared by ImageRequest) ─────────────────────────────


ImageRefRole = Literal["portrait", "scene", "reference", "source"]


@dataclass(frozen=True)
class ImageRef:
    """A reference image fed to an image backend.

    Roles:
      - "portrait"  : character identity reference
      - "scene"     : scene-level reference image
      - "reference" : generic reference (style / subject / environment)
      - "source"    : primary source image for image_edit requests
    """
    role: str
    url: str

    def __post_init__(self) -> None:
        if self.role not in ("portrait", "scene", "reference", "source"):
            raise ValueError(f"ImageRef.role={self.role!r} invalid")
        if not self.url:
            raise ValueError("ImageRef.url is empty")


# ── Segment / Bridge ──────────────────────────────────────────────────────


SegmentMode = Literal["i2v", "r2v"]


@dataclass(frozen=True)
class SegmentRequest:
    """A single segment render (first frame anchored, with optional refs).

    The ``mode`` field is explicit — framework decides whether to run i2v
    (first-only) or r2v (first + soft reference set). Backends MUST honor
    ``mode``; they never auto-infer based on len(refs).
    """
    request_id: str
    prompt: str
    first_url: str
    mode: SegmentMode
    reference_image_urls: tuple[str, ...] = ()
    duration_s: float = 5.0
    seed: int = 0
    output_path: str = ""
    log_dir: str | None = None
    negative_prompt: str = ""
    resolution: str = "1920x1080"
    prompt_extend: bool = False

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError(f"SegmentRequest({self.request_id}) prompt is empty")
        if self.duration_s <= 0:
            raise ValueError(
                f"SegmentRequest({self.request_id}) duration_s={self.duration_s} ≤ 0"
            )
        if self.mode not in ("i2v", "r2v"):
            raise ValueError(
                f"SegmentRequest({self.request_id}) mode={self.mode!r} "
                f"must be 'i2v' or 'r2v'"
            )
        if not self.first_url:
            raise ValueError(f"SegmentRequest({self.request_id}) first_url is empty")


@dataclass(frozen=True)
class BridgeRequest:
    """A chunk-boundary bridge render (first + last frame anchored)."""
    request_id: str
    prompt: str
    first_url: str
    last_url: str
    duration_s: float = 5.0
    seed: int = 0
    output_path: str = ""
    log_dir: str | None = None
    negative_prompt: str = ""
    resolution: str = "1920x1080"
    prompt_extend: bool = False

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError(f"BridgeRequest({self.request_id}) prompt is empty")
        if self.duration_s <= 0:
            raise ValueError(
                f"BridgeRequest({self.request_id}) duration_s={self.duration_s} ≤ 0"
            )
        if not self.first_url:
            raise ValueError(f"BridgeRequest({self.request_id}) first_url is empty")
        if not self.last_url:
            raise ValueError(f"BridgeRequest({self.request_id}) last_url is empty")


# ── Image generation ──────────────────────────────────────────────────────


ImageKind = Literal["portrait", "anchor_image", "location", "prop", "image_edit"]


@dataclass(frozen=True)
class ImageRequest:
    """A single image render request.

    ``kind`` drives dispatch routing (``RECA_RENDER_BACKEND_<KIND>``).
    """
    request_id: str
    kind: ImageKind
    prompt: str
    references: tuple[ImageRef, ...] = ()
    seed: int = 0
    output_path: str = ""
    log_dir: str | None = None
    negative_prompt: str = ""
    resolution: str = "1280x720"

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError(f"ImageRequest({self.request_id}) prompt is empty")
        if self.kind not in ("portrait", "anchor_image", "location", "prop", "image_edit"):
            raise ValueError(
                f"ImageRequest({self.request_id}) kind={self.kind!r} invalid"
            )


# ── Result ────────────────────────────────────────────────────────────────


@dataclass
class VideoResult:
    """Backend-agnostic video render result."""
    request_id: str = ""
    success: bool = True
    output_url: str | None = None
    output_path: str = ""
    rendered_duration_s: float = 0.0
    seed_used: int = 0
    backend_name: str = ""
    error: str = ""
    cost_usd: float = 0.0


# ImageResult mirrors VideoResult (no duration semantics). Kept as a
# distinct symbol so image dispatch can evolve independently if needed.
@dataclass
class ImageResult:
    request_id: str = ""
    success: bool = True
    output_url: str | None = None
    output_path: str = ""
    seed_used: int = 0
    backend_name: str = ""
    error: str = ""
    cost_usd: float = 0.0


__all__ = [
    "ImageRef",
    "ImageRefRole",
    "ImageKind",
    "ImageRequest",
    "ImageResult",
    "SegmentRequest",
    "SegmentMode",
    "BridgeRequest",
    "VideoResult",
]


_ = field  # keep dataclasses.field import explicit
