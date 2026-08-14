"""LTX-2.3 local video backend.

Conforms to ``VideoSegmentBackend``. ``render_segment`` / ``render_bridge``
delegate to ``_runner.run_*``, which shells out to the LTX-2 pipeline CLI
in its own Python env (see ``_runner.py`` for env knobs).

DO NOT import torch / diffusers / huggingface_hub here. The runner is
deliberately subprocess-based so the videorlm main process stays clean.
"""
from __future__ import annotations

from ....interface.capabilities import BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import BridgeRequest, SegmentRequest, VideoResult
from ....interface.segment_backend import (
    ProviderSpec,
    VideoSegmentBackendBase,
)
from . import _runner


class LTXBackend(VideoSegmentBackendBase):
    """Lightricks LTX-2.3 local diffusion video backend."""

    PROVIDER_SPEC = ProviderSpec(
        name="ltx-2.3",
        family="local-diffusion",
        supports_resolutions=("1280x720", "1920x1080"),
        segment_duration_range=(1.0, 25.0),
        bridge_duration_range=(2.0, 8.0),
        max_reference_images=4,
        max_prompt_chars=2000,
        concurrency_env="LTX23_WORKERS",
        default_concurrency=1,
        cost_per_call_usd=0.0,
        cost_per_second_usd=0.0,
        provider="local",
        notes="Local diffusion; runtime placeholder until weights are wired.",
        supports_i2v=True,
        supports_r2v=True,
    )

    def render_segment(self, req: SegmentRequest) -> VideoResult:
        try:
            output = _runner.run_segment(
                request_id=req.request_id,
                prompt=req.prompt,
                first_url=req.first_url,
                reference_image_urls=req.reference_image_urls,
                duration_s=req.duration_s,
                seed=req.seed,
                output_path=req.output_path,
                resolution=req.resolution,
                log_dir=req.log_dir,
                mode=req.mode,
                negative_prompt=req.negative_prompt,
            )
        except (NotImplementedError, FileNotFoundError, RuntimeError, ValueError) as e:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_segment({req.request_id}): {e}"
            ) from e
        return VideoResult(
            request_id=req.request_id,
            success=True,
            output_url=None,
            output_path=output or req.output_path or "",
            rendered_duration_s=float(req.duration_s),
            seed_used=req.seed,
            backend_name=self.PROVIDER_SPEC.name,
        )

    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        try:
            output = _runner.run_bridge(
                request_id=req.request_id,
                prompt=req.prompt,
                first_url=req.first_url,
                last_url=req.last_url,
                duration_s=req.duration_s,
                seed=req.seed,
                output_path=req.output_path,
                resolution=req.resolution,
                log_dir=req.log_dir,
                negative_prompt=req.negative_prompt,
            )
        except (NotImplementedError, FileNotFoundError, RuntimeError, ValueError) as e:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_bridge({req.request_id}): {e}"
            ) from e
        return VideoResult(
            request_id=req.request_id,
            success=True,
            output_url=None,
            output_path=output or req.output_path or "",
            rendered_duration_s=float(req.duration_s),
            seed_used=req.seed,
            backend_name=self.PROVIDER_SPEC.name,
        )


register_backend(LTXBackend.PROVIDER_SPEC.name, LTXBackend())
