"""wan2.6-image backend (DashScope)."""
from __future__ import annotations

import os

from ....interface.capabilities import BackendCapabilities, BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import ImageRequest, ImageResult
from .._platform import platform
from ._common import NO_REF_HINT, collect_image_refs
from ._primitives import generate_image_with_refs


_PLATFORM = platform()


class Wan26ImageBackend:
    """wan2.6-image: multi-modal image generation, requires ≥1 reference."""

    NAME = "wan2.6-image"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name=self.NAME,
            model_family="wan-image",
            supports_kinds=frozenset({"anchor_image", "portrait", "image_edit"}),
            provider=_PLATFORM.provider,
            model_id=self.NAME,
            supports_t2i=False,
            supports_i2i=True,
            supports_first_image=False,
            supports_last_image=False,
            min_duration_s=0.0,
            max_duration_s=0.0,
            duration_granularity_s=0.0,
            supported_resolutions=("1280x720", "1024x1024"),
            max_prompt_chars=2000,
            max_reference_images=4,
            max_concurrency=int(os.environ.get("RECA_WAN26_IMAGE_WORKERS", "8")),
            requests_per_minute=0,
            estimated_cost_per_call_usd=0.005,
        )

    def render(self, request: ImageRequest) -> ImageResult:
        ref_urls = collect_image_refs(request)
        if not ref_urls:
            raise BackendRenderError(
                f"{self.NAME}.render({request.request_id}): {NO_REF_HINT}"
            )
        try:
            url = generate_image_with_refs(
                request.prompt,
                request.output_path or "",
                refs=ref_urls,
                seed=request.seed,
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


register_backend(Wan26ImageBackend.NAME, Wan26ImageBackend())
