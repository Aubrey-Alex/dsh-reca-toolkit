"""happyhorse-1.0-r2v video backend (multi-ref soft target, no first_frame slot).

Conforms to ``VideoSegmentBackend``. happyhorse-r2v takes only
``reference_image`` entries (no first_frame slot); ``first_url`` is fed as
``reference_image[0]`` with a prompt prefix that pins it as the start
frame in the model's instruction.

Two render paths:

  - ``render_segment(req)`` — first frame + reference set (i2v mode drops
    the ref set, r2v mode passes them all).
  - ``render_bridge(req)`` — first + last as ref soft targets.
"""
from __future__ import annotations

import os

from ....interface.capabilities import BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import BridgeRequest, SegmentRequest, VideoResult
from ....interface.segment_backend import (
    ProviderSpec,
    VideoSegmentBackendBase,
)
from .._platform import platform
from ._happyhorse_base import resolution_param, submit_poll_download
from ._r2v_prompt import HAPPYHORSE_R2V_PROMPT_TEMPLATE, prefix_r2v


_PLATFORM = platform()


class HappyHorse10R2VBackend(VideoSegmentBackendBase):
    """happyhorse-1.0-r2v: ordered reference_image entries."""

    PROVIDER_SPEC = ProviderSpec(
        name="happyhorse-1.0-r2v",
        family="happyhorse",
        supports_resolutions=("1280x720", "1920x1080"),
        segment_duration_range=(3.0, 15.0),
        bridge_duration_range=(3.0, 15.0),
        max_reference_images=9,
        max_prompt_chars=2000,
        concurrency_env="RECA_HAPPYHORSE_R2V_WORKERS",
        default_concurrency=24,
        per_key_concurrency=8,
        cost_per_call_usd=0.08,
        cost_per_second_usd=0.008,
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
        if req.mode == "r2v":
            soft = [u for u in req.reference_image_urls if u][:8]
        else:
            soft = []
        return self._do_r2v(
            request_id=req.request_id,
            prompt=prefix_r2v(HAPPYHORSE_R2V_PROMPT_TEMPLATE, req.prompt),
            first_url=req.first_url,
            soft_targets=soft,
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt,
        )

    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        if not req.first_url or not req.last_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_bridge({req.request_id}): "
                f"both first_url and last_url required"
            )
        return self._do_r2v(
            request_id=req.request_id,
            prompt=prefix_r2v(HAPPYHORSE_R2V_PROMPT_TEMPLATE, req.prompt),
            first_url=req.first_url,
            soft_targets=[req.last_url],
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt,
        )

    def _do_r2v(
        self, *, request_id: str, prompt: str, first_url: str,
        soft_targets: list[str], duration_s: float, seed: int,
        output_path: str, log_dir: str | None, resolution: str | None,
        negative_prompt: str = "",
    ) -> VideoResult:
        media: list[dict] = [{"type": "reference_image", "url": first_url}]
        for url in soft_targets[:8]:
            if url:
                media.append({"type": "reference_image", "url": url})
        duration = max(3, int(round(duration_s)))
        params: dict = {
            "resolution": resolution_param(resolution, "1080P"),
            "duration": duration,
            "watermark": False,
            "seed": int(seed or 0),
        }
        ratio = os.environ.get("RECA_HAPPYHORSE_R2V_RATIO", "").strip()
        if ratio:
            params["ratio"] = ratio
        input_block: dict = {"prompt": prompt or "", "media": media}
        if negative_prompt:
            input_block["negative_prompt"] = negative_prompt
        payload = {
            "model": self.PROVIDER_SPEC.name,
            "input": input_block,
            "parameters": params,
        }
        video_url = submit_poll_download(
            payload, output_path or "",
            model_name=self.PROVIDER_SPEC.name, request_id=request_id,
            log_dir=log_dir,
        )
        return VideoResult(
            request_id=request_id,
            success=True,
            output_url=video_url,
            output_path=output_path or "",
            rendered_duration_s=float(duration),
            seed_used=int(seed or 0),
            backend_name=self.PROVIDER_SPEC.name,
        )


register_backend(HappyHorse10R2VBackend.PROVIDER_SPEC.name, HappyHorse10R2VBackend())
