"""happyhorse-1.0-i2v video backend (first frame only, no last-frame API).

Conforms to ``VideoSegmentBackend``. Two render paths:

  - ``render_segment(req)`` with mode="i2v"  — first frame only.
  - ``render_bridge(req)`` — first frame anchor; last_url is dropped because
    the API has no last-frame slot (caller knows: this is for chains where
    last-frame fidelity is sacrificed in exchange for happyhorse cost).
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
from ....._common.trace import trace_event
from .._platform import platform
from ._happyhorse_base import resolution_param, submit_poll_download


_PLATFORM = platform()


class HappyHorse10I2VBackend(VideoSegmentBackendBase):
    """happyhorse-1.0-i2v: first_frame + prompt → video."""

    PROVIDER_SPEC = ProviderSpec(
        name="happyhorse-1.0-i2v",
        family="happyhorse",
        supports_resolutions=("1280x720", "1920x1080"),
        segment_duration_range=(3.0, 15.0),
        bridge_duration_range=(3.0, 15.0),
        max_reference_images=1,
        max_prompt_chars=2000,
        concurrency_env="RECA_HAPPYHORSE_I2V_WORKERS",
        default_concurrency=8,
        per_key_concurrency=8,
        cost_per_call_usd=0.08,
        cost_per_second_usd=0.008,
        provider=_PLATFORM.provider,
        supports_i2v=True,
        supports_r2v=False,
    )

    def render_segment(self, req: SegmentRequest) -> VideoResult:
        if req.mode != "i2v":
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name} supports mode='i2v' only; "
                f"got mode={req.mode!r}. Use happyhorse-1.0-r2v for r2v."
            )
        if not req.first_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_segment({req.request_id}): "
                f"first_url required"
            )
        return self._do_i2v(
            request_id=req.request_id,
            prompt=req.prompt,
            first_url=req.first_url,
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt,
        )

    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        if not req.first_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_bridge({req.request_id}): "
                f"first_url required"
            )
        if req.last_url:
            trace_event(
                "video.degrade", "no_last_frame", log_dir=req.log_dir,
                backend=self.PROVIDER_SPEC.name, request_id=req.request_id,
                reason="happyhorse-i2v has no last_frame API; ignoring last_url",
            )
        return self._do_i2v(
            request_id=req.request_id,
            prompt=req.prompt,
            first_url=req.first_url,
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt,
        )

    def _do_i2v(
        self, *, request_id: str, prompt: str, first_url: str,
        duration_s: float, seed: int, output_path: str,
        log_dir: str | None, resolution: str | None,
        negative_prompt: str = "",
    ) -> VideoResult:
        duration = max(3, int(round(duration_s)))
        input_block: dict = {
            "prompt": prompt or "",
            "media": [{"type": "first_frame", "url": first_url}],
        }
        if negative_prompt:
            input_block["negative_prompt"] = negative_prompt
        payload = {
            "model": self.PROVIDER_SPEC.name,
            "input": input_block,
            "parameters": {
                "resolution": resolution_param(resolution, "1080P"),
                "duration": duration,
                "watermark": False,
                "seed": int(seed or 0),
            },
        }
        ratio = os.environ.get("RECA_HAPPYHORSE_I2V_RATIO", "").strip()
        if ratio:
            payload["parameters"]["ratio"] = ratio
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


register_backend(HappyHorse10I2VBackend.PROVIDER_SPEC.name, HappyHorse10I2VBackend())
