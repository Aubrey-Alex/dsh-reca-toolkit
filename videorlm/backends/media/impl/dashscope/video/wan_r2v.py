"""wan2.7-r2v video backend (first frame + soft reference set).

Conforms to ``VideoSegmentBackend``. Two render paths:

  - ``render_segment(req)`` — first frame + reference_image[s] as soft
    targets (mode = i2v or r2v both supported; for mode=i2v refs are
    ignored and only first_frame is fed).
  - ``render_bridge(req)`` — first_frame + last_url as soft reference.
"""
from __future__ import annotations

from ....interface.capabilities import BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import BridgeRequest, SegmentRequest, VideoResult
from ....interface.segment_backend import (
    ProviderSpec,
    VideoSegmentBackendBase,
)
from .._platform import platform
from ._primitives import generate_r2v_video
from ._r2v_prompt import R2V_PROMPT_TEMPLATE, prefix_r2v


_PLATFORM = platform()


class Wan27R2VBackend(VideoSegmentBackendBase):
    """wan2.7-r2v: first frame + 0..4 reference_image soft targets."""

    PROVIDER_SPEC = ProviderSpec(
        name="wan2.7-r2v",
        family="wan-video",
        supports_resolutions=("1280x720",),
        segment_duration_range=(3.0, 15.0),
        bridge_duration_range=(3.0, 8.0),
        max_reference_images=4,
        max_prompt_chars=2000,
        concurrency_env="RECA_WAN27_R2V_WORKERS",
        default_concurrency=16,
        per_key_concurrency=8,
        cost_per_call_usd=0.12,
        cost_per_second_usd=0.015,
        provider=_PLATFORM.provider,
        supports_i2v=True,
        supports_r2v=True,
    )

    def render_segment(self, req: SegmentRequest) -> VideoResult:
        if not req.first_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_segment({req.request_id}): "
                f"first_url required"
            )
        refs: list[str] = []
        if req.mode == "r2v":
            refs = [u for u in req.reference_image_urls if u][:4]
        # mode == "i2v": refs intentionally ignored — first frame only.
        prefixed = prefix_r2v(R2V_PROMPT_TEMPLATE, req.prompt)
        return self._do_r2v(
            request_id=req.request_id,
            prompt=prefixed,
            first_url=req.first_url,
            ref_urls=refs,
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            negative_prompt=req.negative_prompt,
            prompt_extend=req.prompt_extend,
        )

    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        if not req.first_url or not req.last_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_bridge({req.request_id}): "
                f"both first_url and last_url required"
            )
        prefixed = prefix_r2v(R2V_PROMPT_TEMPLATE, req.prompt)
        return self._do_r2v(
            request_id=req.request_id,
            prompt=prefixed,
            first_url=req.first_url,
            ref_urls=[req.last_url],
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            negative_prompt=req.negative_prompt,
            prompt_extend=req.prompt_extend,
        )

    def _do_r2v(
        self, *, request_id: str, prompt: str, first_url: str,
        ref_urls: list[str], duration_s: float, seed: int,
        output_path: str, log_dir: str | None,
        negative_prompt: str = "", prompt_extend: bool = False,
    ) -> VideoResult:
        duration = max(1, int(round(duration_s)))
        try:
            generate_r2v_video(
                prompt=prompt,
                first_image_url=first_url,
                reference_urls=ref_urls,
                save_path=output_path or "",
                seed=seed,
                duration=duration,
                log_dir=log_dir,
                negative_prompt=negative_prompt or None,
                prompt_extend=prompt_extend,
            )
        except Exception as e:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}({request_id}): "
                f"{type(e).__name__}: {str(e)[:200]}"
            ) from e
        return VideoResult(
            request_id=request_id,
            success=True,
            output_url=None,
            output_path=output_path or "",
            rendered_duration_s=float(duration),
            seed_used=seed,
            backend_name=self.PROVIDER_SPEC.name,
        )


register_backend(Wan27R2VBackend.PROVIDER_SPEC.name, Wan27R2VBackend())
