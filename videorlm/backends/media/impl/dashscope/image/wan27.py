"""wan2.7-image / wan2.7-image-pro backends (DashScope).

Image backends are not under the segment Protocol; they expose a plain
``render(ImageRequest) -> ImageResult`` consumed by ``dispatch_image``.
"""
from __future__ import annotations

import os

from ....interface.capabilities import BackendCapabilities, BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import ImageRequest, ImageResult
from .._platform import platform
from ._common import collect_image_refs, request_size
from ._primitives import generate_image_with_refs, generate_portrait


_PLATFORM = platform()


class Wan27ImageBackend:
    """wan2.7-image: portrait / anchor / edit. Native pure-T2I support."""

    NAME = "wan2.7-image"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name=self.NAME,
            model_family="wan-image",
            supports_kinds=frozenset({
                "portrait", "anchor_image", "image_edit", "location", "prop",
            }),
            provider=_PLATFORM.provider,
            model_id=self.NAME,
            supports_t2i=True,
            supports_i2i=True,
            supports_first_image=False,
            supports_last_image=False,
            min_duration_s=0.0,
            max_duration_s=0.0,
            duration_granularity_s=0.0,
            supported_resolutions=("1280x720", "1024x1024"),
            max_prompt_chars=2000,
            max_reference_images=9,
            max_concurrency=int(os.environ.get("RECA_WAN27_IMAGE_WORKERS", "8")),
            requests_per_minute=0,
            estimated_cost_per_call_usd=0.008,
        )

    def render(self, request: ImageRequest) -> ImageResult:
        ref_urls = collect_image_refs(request)
        try:
            if ref_urls:
                url = generate_image_with_refs(
                    request.prompt,
                    request.output_path or "",
                    refs=ref_urls,
                    seed=request.seed,
                    log_dir=request.log_dir,
                    model=self.NAME,
                    negative_prompt=request.negative_prompt or None,
                )
            else:
                url = generate_portrait(
                    request.prompt,
                    request.output_path or "",
                    seed=request.seed,
                    size=request_size(request),
                    log_dir=request.log_dir,
                    model=self.NAME,
                    negative_prompt=request.negative_prompt or None,
                )
        except Exception as e:
            raise BackendRenderError(
                f"{self.NAME}.render({request.request_id}): "
                f"{type(e).__name__}: {str(e)[:200]}"
            ) from e
        return ImageResult(
            request_id=request.request_id,
            success=True,
            output_url=url,
            output_path=request.output_path or "",
            seed_used=request.seed,
            backend_name=self.NAME,
        )


class Wan27ImageProBackend(Wan27ImageBackend):
    """wan2.7-image-pro: higher-quality variant."""

    NAME = "wan2.7-image-pro"

    def capabilities(self) -> BackendCapabilities:
        caps = super().capabilities()
        data = dict(caps.__dict__)
        data.update({
            "backend_name": self.NAME,
            "model_id": self.NAME,
            "max_concurrency": int(os.environ.get(
                "RECA_WAN27_IMAGE_PRO_WORKERS",
                str(caps.max_concurrency),
            )),
            "estimated_cost_per_call_usd": 0.012,
        })
        return BackendCapabilities(**data)


register_backend(Wan27ImageBackend.NAME, Wan27ImageBackend())
register_backend(Wan27ImageProBackend.NAME, Wan27ImageProBackend())
