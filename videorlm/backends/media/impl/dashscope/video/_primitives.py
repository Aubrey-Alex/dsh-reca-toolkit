"""Video-side DashScope primitives.

Wan FLF2V / R2V submit + poll + download. Each is a thin wrapper
around the dashscope SDK with retries, content-filter seed jitter, and
trace events. Backends call into these from their typed render methods.
"""
from __future__ import annotations

import os

from .._platform import submit_with_key
from ....._common.dashscope_sdk import (
    NEG,
    WAN_I2V,
    download_file,
    log_wan_call,
    poll_task,
    submit_with_retry,
)
from ....._common.trace import trace_event


MAX_CONTENT_RETRIES = int(os.environ.get("RECA_MAX_CONTENT_RETRIES", "3"))

WAN_R2V = "wan2.7-r2v"


# ── wan2.7-i2v FLF2V ───────────────────────────────────────────────────


def generate_i2v_first_last(
    prompt: str,
    first_image_url: str,
    last_image_url: str,
    save_path: str,
    *,
    seed: int,
    duration: int,
    log_dir: str | None = None,
    model: str | None = None,
    negative_prompt: str | None = None,
    prompt_extend: bool = False,
) -> None:
    """Wan 2.7 i2v with first-frame + last-frame anchors.

    ``negative_prompt`` falls back to ``NEG`` when None / empty.
    ``prompt_extend`` is the wan SDK flag (default False).
    """
    used_model = model or WAN_I2V
    used_neg = negative_prompt or NEG
    from dashscope import VideoSynthesis  # lazy: Wan3 HTTP backend does not need the SDK
    last = None
    for attempt in range(MAX_CONTENT_RETRIES):
        jittered = seed + attempt * 1000
        log_wan_call(
            "i2v_flf", used_model, prompt,
            ref_images=[first_image_url, last_image_url],
            seed=jittered, log_dir=log_dir,
        )

        def _call(s=jittered):
            return submit_with_key(lambda api_key: VideoSynthesis.async_call(
                api_key=api_key,
                model=used_model,
                prompt=prompt,
                negative_prompt=used_neg,
                prompt_extend=prompt_extend,
                media=[
                    {"url": first_image_url, "type": "first_frame"},
                    {"url": last_image_url, "type": "last_frame"},
                ],
                duration=duration,
                seed=s,
            ))

        trace_event(
            "wan.primitive", "submit_start", log_dir=log_dir,
            op="i2v_flf", model=used_model, attempt=attempt + 1,
            seed=jittered, duration=duration, save_path=save_path,
        )
        tid = submit_with_retry(_call, "i2v(flf)", used_model)
        trace_event(
            "wan.primitive", "submit_ok", log_dir=log_dir,
            op="i2v_flf", model=used_model, attempt=attempt + 1,
            task_id=tid, save_path=save_path,
        )
        trace_event(
            "wan.primitive", "poll_start", log_dir=log_dir,
            op="i2v_flf", model=used_model, attempt=attempt + 1,
            task_id=tid,
        )
        d = poll_task(tid)
        trace_event(
            "wan.primitive", "poll_result", log_dir=log_dir,
            op="i2v_flf", model=used_model, attempt=attempt + 1,
            task_id=tid, status=d.get("task_status"),
            code=d.get("code"), has_video=bool(d.get("video_url")),
        )
        if d.get("task_status") == "SUCCEEDED" and d.get("video_url"):
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            download_file(d["video_url"], save_path)
            trace_event(
                "wan.primitive", "download_ok", log_dir=log_dir,
                op="i2v_flf", model=used_model, task_id=tid,
                save_path=save_path,
            )
            return
        last = d
        if d.get("code") == "DataInspectionFailed" and attempt < MAX_CONTENT_RETRIES - 1:
            continue
        raise RuntimeError(f"FLF2V failed: {d}")
    raise RuntimeError(f"FLF2V failed: {last}")


# ── wan2.7-r2v ──────────────────────────────────────────────────────────


def generate_r2v_video(
    prompt: str,
    first_image_url: str,
    reference_urls: list[str],
    save_path: str,
    *,
    seed: int,
    duration: int,
    log_dir: str | None = None,
    negative_prompt: str | None = None,
    prompt_extend: bool = False,
) -> None:
    """Wan 2.7 r2v with first_frame + 1..4 reference_image(s).

    Caller is responsible for upstream R2V prompt template prefix. The
    reference_image is a SOFT target — does not pin the last frame.
    ``negative_prompt`` falls back to ``NEG`` when None / empty.
    """
    if not first_image_url:
        raise ValueError("generate_r2v_video: first_image_url required")
    refs = [u for u in (reference_urls or []) if u][:4]
    used_neg = negative_prompt or NEG
    from dashscope import VideoSynthesis  # lazy: Wan3 HTTP backend does not need the SDK
    last = None
    for attempt in range(MAX_CONTENT_RETRIES):
        jittered = seed + attempt * 1000
        log_wan_call(
            "r2v", WAN_R2V, prompt,
            ref_images=[first_image_url, *refs],
            seed=jittered, log_dir=log_dir,
        )

        def _call(s=jittered):
            media = [{"url": first_image_url, "type": "first_frame"}]
            for u in refs:
                media.append({"url": u, "type": "reference_image"})
            return submit_with_key(lambda api_key: VideoSynthesis.async_call(
                api_key=api_key,
                model=WAN_R2V,
                prompt=prompt,
                negative_prompt=used_neg,
                prompt_extend=prompt_extend,
                media=media,
                duration=duration,
                seed=s,
            ))

        trace_event(
            "wan.primitive", "submit_start", log_dir=log_dir,
            op="r2v", model=WAN_R2V, attempt=attempt + 1,
            seed=jittered, duration=duration, save_path=save_path,
            n_refs=len(refs),
        )
        tid = submit_with_retry(_call, "r2v", WAN_R2V)
        trace_event(
            "wan.primitive", "submit_ok", log_dir=log_dir,
            op="r2v", model=WAN_R2V, attempt=attempt + 1,
            task_id=tid, save_path=save_path,
        )
        trace_event(
            "wan.primitive", "poll_start", log_dir=log_dir,
            op="r2v", model=WAN_R2V, attempt=attempt + 1,
            task_id=tid,
        )
        d = poll_task(tid)
        trace_event(
            "wan.primitive", "poll_result", log_dir=log_dir,
            op="r2v", model=WAN_R2V, attempt=attempt + 1,
            task_id=tid, status=d.get("task_status"),
            code=d.get("code"), has_video=bool(d.get("video_url")),
        )
        if d.get("task_status") == "SUCCEEDED" and d.get("video_url"):
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            download_file(d["video_url"], save_path)
            trace_event(
                "wan.primitive", "download_ok", log_dir=log_dir,
                op="r2v", model=WAN_R2V, task_id=tid, save_path=save_path,
            )
            return
        last = d
        if d.get("code") == "DataInspectionFailed" and attempt < MAX_CONTENT_RETRIES - 1:
            continue
        raise RuntimeError(f"R2V failed: {d}")
    raise RuntimeError(f"R2V failed: {last}")
