"""gpt-image-2 / gpt-image-2-pro via the OpenAI Images API.

OpenAI-compatible Images API:
  - T2I  : ``POST /v1/images/generations``
  - I2I  : ``POST /v1/images/edits``       (multipart upload of source image bytes)

Returns ``data[0].b64_json`` (base64-encoded image bytes). We decode and
save to ``request.output_path``. If OSS is configured, the local file is
uploaded and the public URL is returned as ``RenderResult.output_url``;
otherwise ``output_url`` remains None and local-only behavior is preserved.

Routing is env-driven: ``OPENAI_BASE_URL``
points at whichever gateway you want (openai.com direct, a self-hosted
OpenAI-compatible proxy, etc.). Auth env names live on the ``openai``
provider in ``backends/_common/providers/openai.py``.

Concurrency: ``RECA_GPT_IMAGE_2_WORKERS`` env, default 15.
"""
from __future__ import annotations

import base64
import io
import os
import uuid

import httpx

from ....interface.capabilities import BackendCapabilities, BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import ImageRequest, ImageResult
from ....._common.env import env_value
from ....._common.oss_publisher import upload_file as _oss_upload_file
from ....._common.platforms import get_platform, with_key


_PLATFORM = get_platform("openai")


def _open_client(api_key: str | None = None):
    """Construct an OpenAI client bound to whatever gateway
    ``OPENAI_BASE_URL`` points at —
    openai.com direct, a self-hosted proxy, etc.

    Resolution order:
      - api_key  : explicit param (passed in by ``with_key`` middleware
                   when called inside ``render()``) → fall back to
                   ``platform.api_key()`` for non-pooled call sites.
      - base_url : env ``OPENAI_BASE_URL``
                   (+ ``/v1`` appended if missing) → platform default →
                   openai.com when nothing is set.
    """
    from openai import OpenAI

    if api_key is None:
        api_key = _PLATFORM.api_key()
    if not api_key:
        raise BackendRenderError(
            "gpt-image-2: no api key in env "
            f"(checked envs={_PLATFORM.api_key_envs})"
        )
    base_url = (
        env_value("OPENAI_BASE_URL").strip()
        or _PLATFORM.base_url
        or "https://api.openai.com/v1"
    )
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    # max_retries=0: the SDK's silent built-in retries (default 2) re-use
    # the exact same body+headers, so they hit the same Cloudflare-pinned
    # shard as the original call. Disabling them lets the framework's
    # retry_until_exhausted be the *only* retry layer, and each framework
    # attempt mints a fresh X-Client-Request-Id below to break sticky
    # routing.
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


def _save_b64(b64: str, output_path: str) -> None:
    if not output_path:
        raise BackendRenderError("gpt-image-2: output_path required")
    raw = base64.b64decode(b64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(raw)


def _fetch_url(url: str) -> bytes:
    if url and not str(url).startswith(("http://", "https://")):
        with open(url, "rb") as f:
            return f.read()
    rsp = httpx.get(url, timeout=120)
    rsp.raise_for_status()
    return rsp.content


def _normalize_size(size: str | None) -> str:
    """Map our DashScope-style "WxH" to OpenAI-style "WxH" (same syntax;
    just normalize the * separator if any)."""
    if not size:
        return "1024x1024"
    return str(size).replace("*", "x")


class GPTImage2Backend:
    """gpt-image-2 via the OpenAI Images API.

    Returns local output plus an OSS public URL when OSS env is configured.
    Without OSS env, it remains local-only and callers can fall back to
    ``output_path``.
    """

    NAME = "gpt-image-2"
    MODEL_ID = "gpt-image-2"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name=self.NAME,
            model_family="openai-image",
            supports_kinds=frozenset({"anchor_image", "portrait", "image_edit"}),
            provider=_PLATFORM.provider,
            model_id=self.MODEL_ID,
            supports_t2i=True,
            supports_i2i=True,
            supports_first_image=False,
            supports_last_image=False,
            min_duration_s=0.0,
            max_duration_s=0.0,
            duration_granularity_s=0.0,
            # gpt-image-2 size constraint: 16px multiples, 3:1 max ratio,
            # 655_360..8_294_400 total pixels. Common safe set:
            supported_resolutions=(
                "1024x1024", "1536x1024", "1024x1536",
                "2048x2048", "3840x2160", "1280x720", "1024x640",
            ),
            max_prompt_chars=32000,
            # OpenAI 官方:gpt-image-1 / gpt-image-1.5 系列 /v1/images/edits 支持
            # 最多 16 张 image,每张 ≤50MB,png/webp/jpg。见
            # platform.openai.com/docs/api-reference/images/createEdit
            max_reference_images=16,
            max_concurrency=int(os.environ.get("RECA_GPT_IMAGE_2_WORKERS", "15")),
            requests_per_minute=0,
            estimated_cost_per_call_usd=0.04,  # rough; depends on size/quality
            # Cloudflare gateway 524 sticky-shard 长尾 — origin 偶发 >120s
            # 触发 Cloudflare 主动 cut。客户端唯一能做的就是多 retry + 长
            # backoff,期待 OpenAI 端的 sticky 路由把同一个 request_id 漂到
            # 不拥塞的 shard 桶上。15 次 retry × 最长 300s backoff
            # ≈ 最坏 30 分钟阻塞,但比整段 pipeline 死掉好。
            max_retries=int(os.environ.get("RECA_GPT_IMAGE_2_MAX_RETRIES", "16")),
            retry_backoff_base_s=2.0,
            retry_backoff_max_s=float(os.environ.get("RECA_GPT_IMAGE_2_BACKOFF_MAX_S", "300")),
        )

    def render(self, request: ImageRequest) -> ImageResult:
        if not request.output_path:
            raise BackendRenderError(
                f"{self.NAME}.render({request.request_id}): output_path required"
            )
        ref_urls = [r.url for r in request.references if r.url]
        size = _normalize_size(request.resolution)

        # Sticky-shard breaker. Three layered techniques against
        # gateway/Cloudflare 524 long-tail (root cause: same request body +
        # same TCP connection re-routes to the same congested origin shard
        # on every retry):
        #   1. X-Client-Request-Id: fresh uuid per attempt → request body
        #      hash differs, Cloudflare/origin can no longer pin us to one
        #      shard. (Officially documented by OpenAI as the canonical
        #      idempotency-tracking header.)
        #   2. Connection: close → drop the keep-alive TCP, force a new
        #      socket → new Cloudflare edge → potentially new origin pool.
        #   3. stream=True with partial_images=0 → SSE chunked response so
        #      first-byte latency stays under Cloudflare's 100s proxy_read
        #      timeout window. (OpenAI community's documented 524
        #      workaround.) The completed event still carries the final
        #      b64; we ignore partial frames.
        attempt_id = uuid.uuid4().hex
        stream_mode = os.environ.get("RECA_GPT_IMAGE_2_STREAM", "0") == "1"
        extra_headers = {
            "X-Client-Request-Id": attempt_id,
            "Connection": "close",
        }

        def _start_stream(client):
            """Kick off either the T2I or I2I SSE stream, returns the
            stream iterator. Closure over ``ref_urls`` / ``size`` /
            ``request.prompt`` / ``extra_headers``."""
            common = {
                "model": self.MODEL_ID,
                "prompt": request.prompt,
                "size": size,
                "n": 1,
                "extra_headers": extra_headers,
            }
            if stream_mode:
                common.update(stream=True, partial_images=0)
            if not ref_urls:
                return client.images.generate(**common)
            # Multi-image edit. Fetch all refs to bytes; OpenAI SDK takes
            # `image=` as a single file or list of (filename, bytes) tuples.
            # OpenAI /v1/images/edits 支持最多 16 张 image
            # Cap refs to reduce origin GPU inference time (each extra ref
            # costs decode + cross-attn + tighter Cloudflare 120s squeeze).
            # 6 refs has been observed to trigger sticky 524 on the
            # gateway-hk gateway. Default 4 keeps portrait+scene+key prop.
            # Order is set by pipeline _anchor_req: portraits, scene, props
            # — earlier roles are most important for identity grounding.
            _REF_CAP = int(os.environ.get("RECA_GPT_IMAGE_2_REF_CAP", "4"))
            images_payload: list[tuple[str, bytes, str]] = []
            for i, url in enumerate(ref_urls[:_REF_CAP]):
                images_payload.append(
                    (f"ref_{i}.png", _fetch_url(url), "image/png")
                )
            if len(images_payload) == 1:
                img_arg = io.BytesIO(images_payload[0][1])
                img_arg.name = images_payload[0][0]
                common["image"] = img_arg
            else:
                common["image"] = images_payload
            return client.images.edit(**common)

        def _do_call(api_key: str) -> str | None:
            """Pick a key (via with_key middleware), open the OpenAI client
            with it, run the stream to completion, return the final b64."""
            client = _open_client(api_key)
            response = _start_stream(client)
            if not stream_mode:
                data = getattr(response, "data", None) or []
                if data:
                    return getattr(data[0], "b64_json", None)
                return None
            # Consume the SSE stream. Final b64 lives on the *.completed
            # event; partial_images=0 means we expect exactly one such
            # event with no partials.
            b64: str | None = None
            for event in response:
                etype = getattr(event, "type", "")
                if etype in ("image_edit.completed", "image_generation.completed"):
                    b64 = getattr(event, "b64_json", None) or b64
                    break
                # Defensive: some gateways emit only partial frames; the last
                # one IS the final at full quality when partial_images=0.
                if getattr(event, "b64_json", None):
                    b64 = event.b64_json
            return b64

        try:
            # model=self.MODEL_ID ⇒ per-model RateLimiter applies the JSON
            # config (configs/openai_rate_limits.json: 2.0 s Cloudflare-safe
            # interval + max_parallel=8). Without ``model=`` only the
            # KeyPool layer + caps semaphore would gate; the JSON limits
            # would be metadata only.
            b64 = with_key("openai", _do_call, model=self.MODEL_ID)
        except Exception as e:
            raise BackendRenderError(
                f"{self.NAME}.render({request.request_id}): "
                f"{type(e).__name__}: {str(e)[:200]} "
                f"(x_client_request_id={attempt_id})"
            ) from e

        if not b64:
            raise BackendRenderError(
                f"{self.NAME}.render({request.request_id}): "
                f"stream completed without b64_json "
                f"(x_client_request_id={attempt_id})"
            )
        _save_b64(b64, request.output_path)

        # Optional OSS upload so downstream wan/happyhorse calls can use the
        # generated image as a ref. Returns None if OSS env not configured;
        # caller falls back to output_path (local-only) in that case.
        public_url = _oss_upload_file(
            request.output_path,
            prefix=f"{self.NAME}/{request.kind}",
            request_id=request.request_id,
            log_dir=request.log_dir,
        )

        return ImageResult(
            request_id=request.request_id,
            success=True,
            output_url=public_url,           # OSS URL if configured, else None
            output_path=request.output_path or "",
            seed_used=request.seed,
            backend_name=self.NAME,
        )


class GPTImage2ProBackend(GPTImage2Backend):
    """gpt-image-2-pro edit/generation alias exposed by the OpenAI-compatible gateway."""

    NAME = "gpt-image-2-pro"
    MODEL_ID = "gpt-image-2-pro"

    def capabilities(self) -> BackendCapabilities:
        caps = super().capabilities()
        data = dict(caps.__dict__)
        data.update({
            "backend_name": self.NAME,
            "model_id": self.MODEL_ID,
            "estimated_cost_per_call_usd": 0.06,
        })
        return BackendCapabilities(**data)


register_backend(GPTImage2Backend.NAME, GPTImage2Backend())
register_backend(GPTImage2ProBackend.NAME, GPTImage2ProBackend())
