"""qwen-image-2.0-pro backend (DashScope).

Dual-mode: pure T2I via the sync multimodal-generation endpoint, plus
i2i / image-edit via the async multimodal-image endpoint (≤3 refs).

T2I path (0 refs) — verified 2026-05-16 against the qwen-image-2.0-pro
sync endpoint at ``…/aigc/multimodal-generation/generation``. Single
2048×2048 PNG returned inline in ~28 s, 24-hour signed URL. Docs:
https://help.aliyun.com/zh/model-studio/qwen-image-api

i2i path (≥1 ref) — same multimodal endpoint as wan2.x-image, ≤3 input
refs per the qwen-image-edit family server cap.
"""
from __future__ import annotations

import os

from ....interface.capabilities import BackendCapabilities, BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import ImageRequest, ImageResult
from .._platform import platform
from ._common import collect_image_refs, request_size
from ._primitives import generate_image_t2i_qwen


_PLATFORM = platform()


def _t2i_size(request: ImageRequest) -> str:
    """Map ImageRequest.resolution → qwen sync endpoint size string.

    qwen-image-2.0 series requires `WxH` total pixels in [512², 2048²];
    1024x1024 is in-range and matches wan2.7-image-pro's request_size
    default elsewhere in the package, but the qwen sync endpoint also
    accepts 1920x1080, 2048x2048, 1536x2688, etc. We pass through the
    caller's resolution converted to ``W*H`` form.
    """
    return request_size(request, default="1024*1024")


class QwenImage20ProBackend:
    """qwen-image-2.0-pro: T2I (sync endpoint) + i2i / anchor / portrait
    variation (async multimodal-image endpoint, ≤3 refs)."""

    NAME = "qwen-image-2.0-pro"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name=self.NAME,
            model_family="qwen-image",
            supports_kinds=frozenset({"anchor_image", "portrait", "image_edit"}),
            provider=_PLATFORM.provider,
            model_id=self.NAME,
            supports_t2i=True,
            supports_i2i=True,
            supports_first_image=False,
            supports_last_image=False,
            min_duration_s=0.0,
            max_duration_s=0.0,
            duration_granularity_s=0.0,
            # qwen sync T2I officially accepts any WxH between 512² and
            # 2048². Round-tripping wan2.7-image-pro's 1024² + 1280×720
            # presets here so existing callers keep working.
            supported_resolutions=(
                "1024x1024", "1280x720", "1280x1280", "1920x1080",
                "2048x2048", "1536x2688", "1728x2368", "2368x1728",
                "2688x1536",
            ),
            max_prompt_chars=2000,
            # 官方:Qwen image editing 系列 input images 最多 3 张
            # (qwen-image-edit-max / qwen-image-edit-plus / qwen-image-2.0-pro 均同),
            # 见 alibabacloud.com/help/en/model-studio/qwen-image-edit-api。
            max_reference_images=3,
            max_concurrency=int(os.environ.get("RECA_QWEN_IMAGE_WORKERS", "8")),
            requests_per_minute=0,
            estimated_cost_per_call_usd=0.012,
        )

    # Substrings that indicate the server (or its DataInspection middleware)
    # rejected the prompt/refs for moderation reasons rather than for a
    # transient/network/quota issue. When matched, the qwen backend falls
    # back to wan2.7-image-pro for the SAME request — explicit, single hop,
    # logged loudly so the result is auditable.
    _MODERATION_MARKERS = (
        "datainspectionfailed", "data_inspection_failed",
        "data inspection failed", "input data may contain inappropriate",
        "content_filter", "content_policy", "image_risk",
        "greennet", "green net", "moderation", "inappropriate content",
        "violence", "敏感", "审核未通过", "内容审核",
    )

    def _is_moderation_failure(self, exc: BaseException) -> bool:
        s = f"{type(exc).__name__} {exc}".lower()
        return any(m in s for m in self._MODERATION_MARKERS)

    @staticmethod
    def _fallback_chain() -> list[str]:
        """Comma-separated backend names from ``RECA_IMAGE_FALLBACK_CHAIN``.

        Default: ``wan2.7-image-pro,gpt-image-2`` — i.e. on a qwen-image-2.0-pro
        content-moderation rejection, try wan2.7-image-pro first (same provider,
        permissive on most non-violent fantasy); if that also rejects, try
        gpt-image-2 (different provider, different filter). Empty string
        disables fallback (raise loud on the first moderation hit).
        """
        raw = os.environ.get(
            "RECA_IMAGE_FALLBACK_CHAIN", "wan2.7-image-pro,gpt-image-2"
        ).strip()
        return [s.strip() for s in raw.split(",") if s.strip()]

    def render(self, request: ImageRequest) -> ImageResult:
        ref_urls = collect_image_refs(request)
        size = _t2i_size(request)
        # Both T2I (refs=[]) and i2i (1–3 refs) go through the same sync
        # multimodal-generation endpoint. The async multimodal-image
        # endpoint that ``generate_image_with_refs`` uses doesn't accept
        # qwen-image-2.0-pro at all (returns 400 "url error" regardless of
        # whether refs are local paths or signed OSS URLs).
        try:
            url = generate_image_t2i_qwen(
                request.prompt,
                request.output_path or "",
                seed=request.seed,
                size=size,
                log_dir=request.log_dir,
                model=self.NAME,
                negative_prompt=request.negative_prompt or None,
                refs=ref_urls or None,
            )
        except Exception as e:
            # Content-moderation fallback chain. Walks the configured backends
            # in order; each is loud-logged with reason. If every chain step
            # also rejects, raises the FINAL error so failures stay visible.
            if self._is_moderation_failure(e):
                chain = self._fallback_chain()
                from ....interface.registry import get_backend
                last_exc: BaseException = e
                primary = self.NAME
                for alt_name in chain:
                    if alt_name == primary:  # avoid self-loop
                        continue
                    print(
                        f"[{primary}] {request.request_id}: content-moderation "
                        f"rejection on {primary if last_exc is e else last_exc_backend} "
                        f"— falling back to {alt_name} "
                        f"(reason: {type(last_exc).__name__}: {str(last_exc)[:140]})",
                        flush=True,
                    )
                    try:
                        result = get_backend(alt_name).render(request)
                        result.backend_name = alt_name
                        return result
                    except Exception as e2:
                        last_exc = e2
                        last_exc_backend = alt_name  # noqa: F841 (used in next loop iter print)
                        if not self._is_moderation_failure(e2):
                            # Non-moderation failure on the fallback target
                            # (network / quota / etc.). Don't keep walking
                            # the chain — re-raise so retry layer can backoff.
                            raise BackendRenderError(
                                f"{primary} fallback={alt_name}.render({request.request_id}): "
                                f"{type(e2).__name__}: {str(e2)[:200]}"
                            ) from e2
                # Every chain step rejected on moderation — raise the last one.
                raise BackendRenderError(
                    f"{primary} + fallbacks {chain} all moderation-rejected "
                    f"({request.request_id}): {type(last_exc).__name__}: {str(last_exc)[:200]}"
                ) from last_exc
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


register_backend(QwenImage20ProBackend.NAME, QwenImage20ProBackend())


class QwenImage20Backend(QwenImage20ProBackend):
    """qwen-image-2.0 (non-pro accelerated): same sync endpoint as the Pro
    variant, but the throughput limit is **2 req/s** vs Pro's **2 req/min**
    (60× higher headroom). Slightly lower fidelity, but for batch runs
    where the validator-driven repair burst would otherwise hammer the Pro
    cap into 83 %+ failure rate (verified 2026-05-16: 80 rate_limit + 67
    other errors in 177 calls), this is the right default.

    Capabilities are inherited; only the registered NAME differs so callers
    select via ``RECA_RENDER_BACKEND_*=qwen-image-2.0``.
    """

    NAME = "qwen-image-2.0"


register_backend(QwenImage20Backend.NAME, QwenImage20Backend())
