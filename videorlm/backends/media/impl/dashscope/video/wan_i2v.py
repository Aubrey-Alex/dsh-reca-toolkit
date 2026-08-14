"""wan2.7-i2v video backend (native FLF2V semantics on DashScope).

Conforms to ``VideoSegmentBackend``. Two render paths:

  - ``render_segment(req)`` with ``mode="i2v"`` — first frame only.
    (``mode="r2v"`` is rejected; use ``wan2.7-r2v`` for r2v.)
  - ``render_bridge(req)`` — first + last frame (native FLF2V).
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
from ._primitives import generate_i2v_first_last


_PLATFORM = platform()


class Wan27I2VBackend(VideoSegmentBackendBase):
    """wan2.7-i2v: first frame + optional last frame.

    Use ``mode="i2v"`` for first-only; bridge uses both anchors.
    """

    PROVIDER_SPEC = ProviderSpec(
        name="wan2.7-i2v",
        family="wan-video",
        supports_resolutions=("1280x720",),
        segment_duration_range=(3.0, 15.0),
        bridge_duration_range=(3.0, 8.0),
        max_reference_images=2,
        max_prompt_chars=2000,
        concurrency_env="RECA_WAN27_I2V_WORKERS",
        default_concurrency=8,
        per_key_concurrency=8,
        cost_per_call_usd=0.10,
        cost_per_second_usd=0.012,
        provider=_PLATFORM.provider,
        supports_i2v=True,
        supports_r2v=False,
    )

    # ── segment / bridge ──────────────────────────────────────────────

    def render_segment(self, req: SegmentRequest) -> VideoResult:
        # wan2.7-i2v API requires BOTH first and last frames at the wire level
        # (it is a native FLF2V endpoint, not an i2v-only endpoint). Pure i2v
        # segment generation should route to happyhorse-1.0-i2v instead; if
        # framework picks wan2.7-i2v for a segment, it must explicitly opt in
        # by setting RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST=1, which then
        # synthesizes a near-static clip by reusing first_url as last_url.
        if req.mode != "i2v":
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name} supports mode='i2v' only; "
                f"got mode={req.mode!r}. Use wan2.7-r2v for r2v."
            )
        if not req.first_url:
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_segment({req.request_id}): "
                f"first_url required for i2v mode"
            )
        if os.environ.get("RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST", "").strip() not in ("1", "true", "yes"):
            raise BackendRenderError(
                f"{self.PROVIDER_SPEC.name}.render_segment({req.request_id}): "
                f"wan2.7-i2v is a first+last FLF2V endpoint; pure i2v "
                f"segment generation is not its native shape. Route this "
                f"call to happyhorse-1.0-i2v, or set "
                f"RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST=1 to reuse "
                f"first_url as last_url (yields a near-static clip)."
            )
        return self._do_i2v(
            request_id=req.request_id,
            prompt=req.prompt,
            first_url=req.first_url,
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
                f"both first_url and last_url required (got first="
                f"{bool(req.first_url)} last={bool(req.last_url)})"
            )
        return self._do_flf2v(
            request_id=req.request_id,
            prompt=req.prompt,
            first_url=req.first_url,
            last_url=req.last_url,
            duration_s=req.duration_s,
            seed=req.seed,
            output_path=req.output_path,
            log_dir=req.log_dir,
            negative_prompt=req.negative_prompt,
            prompt_extend=req.prompt_extend,
        )

    # ── primitives ────────────────────────────────────────────────────

    def _do_i2v(
        self, *, request_id: str, prompt: str, first_url: str,
        duration_s: float, seed: int, output_path: str,
        log_dir: str | None, negative_prompt: str = "",
        prompt_extend: bool = False,
    ) -> VideoResult:
        # wan2.7-i2v requires both first + last frames at the API level — for
        # first-only segment generation we reuse first_url as last_url, which
        # yields a near-static clip; framework agent decides whether they want
        # this path or want to route segment_i2v to a backend with a real
        # first-only API (e.g. happyhorse-1.0-i2v).
        return self._do_flf2v(
            request_id=request_id,
            prompt=prompt,
            first_url=first_url,
            last_url=first_url,
            duration_s=duration_s,
            seed=seed,
            output_path=output_path,
            log_dir=log_dir,
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
        )

    def _do_flf2v(
        self, *, request_id: str, prompt: str, first_url: str, last_url: str,
        duration_s: float, seed: int, output_path: str,
        log_dir: str | None,
        negative_prompt: str = "", prompt_extend: bool = False,
    ) -> VideoResult:
        duration = max(1, int(round(duration_s)))
        try:
            generate_i2v_first_last(
                prompt=prompt,
                first_image_url=first_url,
                last_image_url=last_url,
                save_path=output_path or "",
                seed=seed,
                duration=duration,
                log_dir=log_dir,
                model=self.PROVIDER_SPEC.name,
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


register_backend(Wan27I2VBackend.PROVIDER_SPEC.name, Wan27I2VBackend())
