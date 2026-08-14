"""Image-side DashScope primitives.

T2I / multimodal-image submit + poll + download. Used by every image
backend in this package; each backend just shapes its payload + model id.
"""
from __future__ import annotations

import os
from http import HTTPStatus
from types import SimpleNamespace

import httpx

from .._platform import submit_with_key
from ....._common.dashscope_sdk import (
    DEFAULT_RES,
    NEG,
    WANX_T2I,
    download_file,
    log_wan_call,
    poll_image_task,
    submit_with_retry,
)
from ....._common.trace import trace_event


MAX_CONTENT_RETRIES = int(os.environ.get("RECA_MAX_CONTENT_RETRIES", "3"))

_DEFAULT_IMAGE_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
)
_ENDPOINT_BY_MODEL: dict[str, str] = {}


def _endpoint_for(model: str) -> str:
    return _ENDPOINT_BY_MODEL.get(model, _DEFAULT_IMAGE_ENDPOINT)


def _image_synthesis():
    from dashscope import ImageSynthesis
    return ImageSynthesis


_QWEN_IMAGE_SYNC_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)


def generate_image_t2i_qwen(
    prompt: str,
    save_path: str,
    *,
    seed: int,
    size: str = "2048*2048",
    log_dir: str | None = None,
    model: str = "qwen-image-2.0-pro",
    negative_prompt: str | None = None,
    prompt_extend: bool = False,
    refs: list[str] | None = None,
    timeout_s: float = 240.0,
) -> str:
    """T2I / i2i via qwen-image-2.0 series **sync** multimodal-generation endpoint.

    Single endpoint handles both modes:
      - ``refs=[]`` → pure text-to-image (one ``{"text": prompt}`` content block)
      - ``refs=[url, ...]`` → image edit / i2i (1–3 ``{"image": url}`` blocks
        prepended to the text). Image content rules per the qwen-image-edit
        docs: HTTP(S) URL or ``data:image/...;base64,...`` form, ≤10MB,
        formats JPG/JPEG/PNG/BMP/TIFF/WEBP/GIF.

    NB: This is the sync endpoint at ``…/aigc/multimodal-generation/generation``;
    the response carries the image URL inline (24h expiry). Distinct from
    ``_multimodal_image_submit`` (the async ``…/image-generation/generation``
    endpoint) which rejects 0-image input AND silently 400s on
    qwen-image-2.0-pro refs with "url error".

    Docs:
      T2I  https://help.aliyun.com/zh/model-studio/qwen-image-api
      i2i  https://help.aliyun.com/zh/model-studio/qwen-image-edit-api
    """
    refs = refs or []
    content: list[dict] = [{"image": url} for url in refs if url]
    content.append({"text": prompt})
    body: dict = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        },
        "parameters": {
            "negative_prompt": negative_prompt or NEG,
            "prompt_extend": bool(prompt_extend),
            "watermark": False,
            "size": size,
            "n": 1,
            "seed": int(seed),
        },
    }
    last_payload: dict | None = None
    op = "qwen_sync_i2i" if refs else "qwen_sync_t2i"
    for attempt in range(MAX_CONTENT_RETRIES):
        log_wan_call(
            op, model, prompt,
            ref_images=refs, seed=seed + attempt * 1000, log_dir=log_dir,
        )
        body["parameters"]["seed"] = int(seed) + attempt * 1000
        trace_event(
            "wan.primitive", "submit_start", log_dir=log_dir,
            op=op, model=model, attempt=attempt + 1,
            save_path=save_path,
        )

        def _call(api_key: str) -> SimpleNamespace:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            try:
                rsp = httpx.post(
                    _QWEN_IMAGE_SYNC_ENDPOINT, headers=headers,
                    json=body, timeout=timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                return SimpleNamespace(
                    status_code=599, output=SimpleNamespace(task_id=None),
                    code=type(exc).__name__, message=str(exc)[:240],
                    payload={},
                )
            try:
                data = rsp.json()
            except Exception:
                data = {}
            return SimpleNamespace(
                status_code=rsp.status_code,
                output=SimpleNamespace(task_id=data.get("request_id")),
                code=data.get("code") or ("OK" if rsp.status_code == HTTPStatus.OK else f"HTTP_{rsp.status_code}"),
                message=(data.get("message") or "")[:240],
                payload=data,
            )

        rsp = submit_with_key(_call)
        data = getattr(rsp, "payload", {}) or {}
        last_payload = data
        trace_event(
            "wan.primitive", "submit_done", log_dir=log_dir,
            op=op, model=model, attempt=attempt + 1,
            status=rsp.status_code, code=rsp.code,
        )
        if rsp.status_code == HTTPStatus.OK:
            url = _extract_image_url(data.get("output") or {})
            if url:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                download_file(url, save_path)
                trace_event(
                    "wan.primitive", "download_ok", log_dir=log_dir,
                    op=op, model=model, save_path=save_path,
                )
                return str(url)
        # Content-moderation retry (same convention as wanx-v1 / multimodal-image flow)
        if (data.get("code") == "DataInspectionFailed"
                and attempt < MAX_CONTENT_RETRIES - 1):
            continue
        # Other HTTP errors — fail fast unless retry budget remains
        if rsp.status_code != HTTPStatus.OK and attempt < MAX_CONTENT_RETRIES - 1:
            continue
        raise RuntimeError(
            f"qwen sync t2i ({model}) failed: "
            f"status={rsp.status_code} code={rsp.code} msg={rsp.message}"
        )
    raise RuntimeError(f"qwen sync t2i ({model}) failed: {last_payload}")


def _multimodal_image_submit(
    *, prompt: str, refs: list[str], size: str, seed: int, model: str,
) -> SimpleNamespace:
    """Raw HTTP submit for multimodal image endpoint."""
    content: list[dict] = [{"image": url, "type": "image"} for url in refs if url]
    content.append({"text": prompt, "type": "text"})
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {
            "size": size,
            "n": 1,
            "watermark": False,
            "seed": seed,
        },
    }

    def _call(api_key: str) -> SimpleNamespace:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        try:
            rsp = httpx.post(_endpoint_for(model), headers=headers, json=payload, timeout=120)
        except Exception as e:  # noqa: BLE001
            return SimpleNamespace(
                status_code=599,
                output=SimpleNamespace(task_id=None),
                code=type(e).__name__,
                message=str(e)[:200],
            )
        try:
            data = rsp.json()
        except Exception:  # noqa: BLE001
            data = {}
        if rsp.status_code == HTTPStatus.OK:
            out = data.get("output") or {}
            task_id = out.get("task_id")
            return SimpleNamespace(
                status_code=HTTPStatus.OK,
                output=SimpleNamespace(task_id=task_id),
                code="OK",
                message="",
            )
        return SimpleNamespace(
            status_code=rsp.status_code,
            output=SimpleNamespace(task_id=None),
            code=data.get("code") or f"HTTP_{rsp.status_code}",
            message=(data.get("message") or "")[:240],
        )

    return submit_with_key(_call)


def _extract_image_url(payload: dict) -> str | None:
    """Multimodal `choices` envelope, with legacy `results` fallback."""
    choices = payload.get("choices") or []
    if choices and isinstance(choices, list):
        first = choices[0]
        message = (first or {}).get("message") if isinstance(first, dict) else None
        content = (message or {}).get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                url = part.get("image") or part.get("url")
                if url:
                    return str(url)
    results = payload.get("results") or []
    if results and isinstance(results, list):
        first = results[0]
        if isinstance(first, dict):
            url = first.get("url")
            if url:
                return str(url)
    return None


def generate_image_with_refs(
    prompt: str,
    save_path: str,
    *,
    refs: list[str],
    seed: int,
    size: str = DEFAULT_RES,
    log_dir: str | None = None,
    model: str | None = None,
    negative_prompt: str | None = None,
) -> str:
    """T2I with reference images (multimodal). Returns the source URL.

    1-4 ref URLs required (server rejects 0-image input). Pure T2I → use
    ``generate_portrait`` (wanx-v1) instead.

    ``negative_prompt`` is currently passed only via the multimodal endpoint
    payload — DashScope's multimodal-image endpoint does not have a top-level
    negative_prompt field; if needed, prepend "negative: …" or use the
    image-side params.size / watermark — kept for API parity with video calls.
    """
    if not model:
        raise ValueError(
            "generate_image_with_refs requires explicit model= argument; "
            "no env-var fallback. Caller (image backend) must pass model=self.NAME."
        )
    used_model = model
    clean_refs = [str(r).strip() for r in (refs or []) if str(r).strip()]
    if not clean_refs:
        raise ValueError(
            f"generate_image_with_refs requires 1-4 ref URLs (model={used_model} "
            "rejects 0-image input). For pure T2I use generate_portrait (wanx-v1)."
        )
    last = None
    for attempt in range(MAX_CONTENT_RETRIES):
        jittered = seed + attempt * 1000
        log_wan_call(
            "t2i_refs", used_model, prompt,
            ref_images=clean_refs, seed=jittered, log_dir=log_dir,
        )

        def _call(s=jittered, m=used_model):
            return _multimodal_image_submit(
                prompt=prompt, refs=clean_refs, size=size, seed=s, model=m,
            )

        trace_event(
            "wan.primitive", "submit_start", log_dir=log_dir,
            op="t2i_refs", model=used_model, attempt=attempt + 1,
            seed=jittered, save_path=save_path, n_refs=len(clean_refs),
        )
        tid = submit_with_retry(_call, "t2i_refs", used_model)
        trace_event(
            "wan.primitive", "submit_ok", log_dir=log_dir,
            op="t2i_refs", model=used_model, attempt=attempt + 1,
            task_id=tid, save_path=save_path,
        )
        trace_event(
            "wan.primitive", "poll_start", log_dir=log_dir,
            op="t2i_refs", model=used_model, attempt=attempt + 1,
            task_id=tid,
        )
        d = poll_image_task(tid)
        trace_event(
            "wan.primitive", "poll_result", log_dir=log_dir,
            op="t2i_refs", model=used_model, attempt=attempt + 1,
            task_id=tid, status=d.get("task_status"),
            code=d.get("code"), has_image=bool(_extract_image_url(d)),
        )
        if d.get("task_status") == "SUCCEEDED":
            url = _extract_image_url(d)
            if url:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                download_file(url, save_path)
                trace_event(
                    "wan.primitive", "download_ok", log_dir=log_dir,
                    op="t2i_refs", model=used_model, task_id=tid,
                    save_path=save_path,
                )
                return url
        last = d
        if d.get("code") == "DataInspectionFailed" and attempt < MAX_CONTENT_RETRIES - 1:
            continue
        raise RuntimeError(f"T2I({used_model}) failed: {d}")
    raise RuntimeError(f"T2I({used_model}) failed: {last}")


def generate_portrait(
    prompt: str,
    save_path: str,
    *,
    seed: int,
    size: str = "1024*1024",
    log_dir: str | None = None,
    model: str | None = None,
    negative_prompt: str | None = None,
) -> str:
    """Pure T2I (no refs). Returns source URL.

    Default model: ``WANX_T2I`` (wanx-v1). Modern image backends with pure-T2I
    support submit their own model id through the multimodal endpoint with 0 refs.
    ``negative_prompt`` falls back to ``NEG`` when None / empty (only applied
    on the wanx-v1 ImageSynthesis path; multimodal endpoint does not expose it).
    """
    used_model = model or WANX_T2I
    used_neg = negative_prompt or NEG
    last: dict | None = None
    for attempt in range(MAX_CONTENT_RETRIES):
        jittered = seed + attempt * 1000
        log_wan_call(
            "t2i_pure", used_model, prompt,
            ref_images=[], seed=jittered, log_dir=log_dir,
        )

        def _call(s=jittered, m=used_model):
            if m == WANX_T2I:
                return submit_with_key(lambda api_key: _image_synthesis().async_call(
                    api_key=api_key, model=m,
                    prompt=prompt, negative_prompt=used_neg,
                    n=1, size=size, seed=s,
                ))
            return _multimodal_image_submit(
                prompt=prompt, refs=[], size=size, seed=s, model=m,
            )

        trace_event(
            "wan.primitive", "submit_start", log_dir=log_dir,
            op="t2i_pure", model=used_model, attempt=attempt + 1,
            seed=jittered, save_path=save_path,
        )
        tid = submit_with_retry(_call, "t2i", rate_key=used_model)
        trace_event(
            "wan.primitive", "submit_ok", log_dir=log_dir,
            op="t2i_pure", model=used_model, attempt=attempt + 1,
            task_id=tid, save_path=save_path,
        )
        trace_event(
            "wan.primitive", "poll_start", log_dir=log_dir,
            op="t2i_pure", model=used_model, attempt=attempt + 1,
            task_id=tid,
        )
        d = poll_image_task(tid)
        image_url = _extract_image_url(d)
        trace_event(
            "wan.primitive", "poll_result", log_dir=log_dir,
            op="t2i_pure", model=used_model, attempt=attempt + 1,
            task_id=tid, status=d.get("task_status"),
            code=d.get("code"), has_results=bool(d.get("results")),
            has_image=bool(image_url),
        )
        if d.get("task_status") == "SUCCEEDED":
            if image_url:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                download_file(image_url, save_path)
                trace_event(
                    "wan.primitive", "download_ok", log_dir=log_dir,
                    op="t2i_pure", model=used_model, task_id=tid,
                    save_path=save_path,
                )
                return str(image_url)
        last = d
        if d.get("code") == "DataInspectionFailed" and attempt < MAX_CONTENT_RETRIES - 1:
            continue
        raise RuntimeError(f"portrait T2I({used_model}) failed: {d}")
    raise RuntimeError(f"portrait T2I({used_model}) failed: {last}")
