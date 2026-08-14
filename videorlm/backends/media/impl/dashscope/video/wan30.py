"""Wan 3.0 video backend using the Alibaba Model Studio async API.

The request shape follows the working AVM client in ``/mnt/cpfs02/akide/us/haoming/avm``:

  POST /api/v1/services/aigc/video-generation/video-synthesis
  GET  /api/v1/tasks/{task_id}

Local reference media is staged through DashScope's temporary OSS policy API so
the provider can resolve it with ``X-DashScope-OssResourceResolve: enable``.
The ReCA planner, retry policy, validators, and concatenation remain unchanged.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ....interface.capabilities import BackendRenderError
from ....interface.registry import register_backend
from ....interface.requests import BridgeRequest, SegmentRequest, VideoResult
from ....interface.segment_backend import ProviderSpec, VideoSegmentBackendBase


_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
_DEFAULT_MODEL = "wan3.0-video"
_CREATE_PATH = "/services/aigc/video-generation/video-synthesis"
_TASK_PATH = "/tasks/{task_id}"
_UPLOAD_PATH = "/uploads"
_TERMINAL = {"succeeded", "success", "completed", "failed", "cancelled", "canceled", "expired"}
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _api_key() -> str:
    for name in ("RECA_WAN30_API_KEY", "DASHSCOPE_API_KEY", "DASHSCOPE_API_KEYS"):
        value = os.environ.get(name, "").strip()
        if value:
            return value.split(",", 1)[0].strip()
    raise RuntimeError("Wan 3.0 API key is missing; set RECA_WAN30_API_KEY or DASHSCOPE_API_KEY")


def _base_url() -> str:
    return os.environ.get("RECA_WAN30_BASE_URL", "").strip().rstrip("/") or _DEFAULT_BASE_URL


def _model() -> str:
    return os.environ.get("RECA_WAN30_MODEL", "").strip() or _DEFAULT_MODEL


def _resolution(value: str | None) -> str:
    raw = (value or os.environ.get("RECA_WAN30_RESOLUTION", "720P")).strip()
    return {"1280x720": "720P", "1920x1080": "1080P", "720p": "720P", "1080p": "1080P"}.get(raw, raw)


def _url(path: str) -> str:
    return f"{_base_url()}/{path.lstrip('/')}"


def _headers(*, resolve_oss: bool = False) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    if resolve_oss:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    return headers


def _json_request(method: str, url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    timeout = float(os.environ.get("RECA_WAN30_REQUEST_TIMEOUT_S", "180"))
    try:
        response = httpx.request(
            method,
            url,
            headers=_headers(resolve_oss=bool(payload and _contains_oss(payload))),
            json=payload,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Wan 3.0 network request failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text[:4000]
        if response.status_code in _TRANSIENT_STATUS:
            raise _TransientWanError(f"HTTP {response.status_code}: {detail}")
        raise RuntimeError(f"Wan 3.0 HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Wan 3.0 returned non-JSON data") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Wan 3.0 returned a non-object JSON response")
    return data


class _TransientWanError(RuntimeError):
    pass


def _contains_oss(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("oss://")
    if isinstance(value, dict):
        return any(_contains_oss(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_oss(item) for item in value)
    return False


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"DashScope upload policy missing {key!r}")
    return value


def _stage_local_media(path: Path, *, model: str, cache_dir: Path) -> str:
    """Upload one local image/video and return its temporary ``oss://`` URI."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "wan30_staged_media.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    key = str(path.resolve())
    stat = path.stat()
    cached = cache.get(key)
    if (
        isinstance(cached, dict)
        and cached.get("model") == model
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("uri"), str)
    ):
        return cached["uri"]

    policy = _json_request(
        "GET",
        _url(os.environ.get("RECA_WAN30_UPLOAD_PATH", _UPLOAD_PATH))
        + f"?action=getPolicy&model={httpx.QueryParams({'model': model})['model']}",
        payload=None,
    )
    policy_data = policy.get("data") if isinstance(policy.get("data"), dict) else policy
    if not isinstance(policy_data, dict):
        raise RuntimeError("DashScope upload policy response has no data object")
    upload_dir = _required_string(policy_data, "upload_dir").strip("/")
    upload_host = _required_string(policy_data, "upload_host")
    object_key = f"{upload_dir}/{path.name}"
    fields = {
        "OSSAccessKeyId": _required_string(policy_data, "oss_access_key_id"),
        "Signature": _required_string(policy_data, "signature"),
        "policy": _required_string(policy_data, "policy"),
        "x-oss-object-acl": _required_string(policy_data, "x_oss_object_acl"),
        "x-oss-forbid-overwrite": _required_string(policy_data, "x_oss_forbid_overwrite"),
        "key": object_key,
        "success_action_status": "200",
    }
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    timeout = float(os.environ.get("RECA_WAN30_REQUEST_TIMEOUT_S", "180"))
    try:
        with path.open("rb") as handle:
            response = httpx.post(
                upload_host,
                data=fields,
                files={"file": (path.name, handle, mime)},
                timeout=timeout,
            )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"DashScope temporary upload failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code >= 300:
        raise RuntimeError(f"DashScope temporary upload HTTP {response.status_code}: {response.text[:2000]}")
    uri = f"oss://{object_key}"
    cache[key] = {"uri": uri, "model": model, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return uri


def _media_url(value: str, *, model: str, cache_dir: Path) -> str:
    raw = str(value or "").strip()
    if raw.startswith(("http://", "https://", "oss://", "data:")):
        return raw
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"Wan 3.0 reference does not exist: {raw}")
    if os.environ.get("RECA_WAN30_STAGE_LOCAL_MEDIA", "1").strip().lower() in {"0", "false", "no"}:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return "data:" + (mimetypes.guess_type(path.name)[0] or "image/png") + ";base64," + __import__("base64").b64encode(path.read_bytes()).decode("ascii")
        raise RuntimeError("Wan 3.0 local video references require temporary OSS staging")
    return _stage_local_media(path, model=model, cache_dir=cache_dir)


def _task_id(response: dict[str, Any]) -> str:
    output = response.get("output")
    for value in (response.get("task_id"), response.get("id"), output.get("task_id") if isinstance(output, dict) else None, output.get("id") if isinstance(output, dict) else None):
        if isinstance(value, str) and value:
            return value
    raise RuntimeError(f"Wan 3.0 create response has no task id: {response}")


def _status(response: dict[str, Any]) -> str:
    output = response.get("output")
    values = [response.get("task_status"), response.get("status")]
    if isinstance(output, dict):
        values.extend([output.get("task_status"), output.get("status")])
    for value in values:
        if isinstance(value, str) and value:
            return value.lower()
    return ""


def _video_url(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("video_url", "url", "output_url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                return candidate
        for item in value.values():
            found = _video_url(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _video_url(item)
            if found:
                return found
    return None


def _poll(task_id: str) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(os.environ.get("RECA_WAN30_TASK_TIMEOUT_S", "3600"))
    interval = max(0.5, float(os.environ.get("RECA_WAN30_POLL_INTERVAL_S", "5")))
    log_interval = max(5.0, float(os.environ.get("RECA_WAN30_STATUS_LOG_INTERVAL_S", "30")))
    last_status = ""
    last_log = 0.0
    while True:
        try:
            response = _json_request("GET", _url(os.environ.get("RECA_WAN30_STATUS_PATH", _TASK_PATH).format(task_id=task_id)))
        except _TransientWanError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Wan 3.0 task {task_id} polling timed out")
            time.sleep(min(interval, 30.0))
            continue
        status = _status(response)
        now = time.monotonic()
        if status != last_status or now - last_log >= log_interval:
            print(
                f"[wan30] task={task_id} status={status or 'unknown'} "
                f"elapsed={now - started:.0f}s",
                flush=True,
            )
            last_status = status
            last_log = now
        if status in _TERMINAL:
            return response
        if now >= deadline:
            raise TimeoutError(f"Wan 3.0 task {task_id} did not finish within timeout")
        time.sleep(interval)


def _download(url: str, destination: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    timeout = float(os.environ.get("RECA_WAN30_REQUEST_TIMEOUT_S", "180"))
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Wan 3.0 result download failed: {type(exc).__name__}: {exc}") from exc
    temporary.replace(path)


def _generate(*, request_id: str, prompt: str, media: list[dict[str, str]], duration_s: float, seed: int, output_path: str, log_dir: str | None, resolution: str | None, negative_prompt: str) -> str:
    model = _model()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(log_dir or output_path).resolve().parent
    staged_media = [{"type": item["type"], "url": _media_url(item["url"], model=model, cache_dir=cache_dir)} for item in media]
    duration = max(2, min(15, int(round(duration_s))))
    payload = {
        "model": model,
        "input": {"prompt": prompt, "media": staged_media},
        "parameters": {
            "duration": duration,
            "resolution": _resolution(resolution),
            "ratio": os.environ.get("RECA_WAN30_RATIO", "16:9"),
            "prompt_extend": os.environ.get("RECA_WAN30_PROMPT_EXTEND", "false").lower() in {"1", "true", "yes"},
            "watermark": os.environ.get("RECA_WAN30_WATERMARK", "false").lower() in {"1", "true", "yes"},
            "seed": int(seed or 0),
        },
    }
    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt
    state_path = Path(output_path).with_suffix(Path(output_path).suffix + ".wan30.json")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    task_id = state.get("task_id") if isinstance(state, dict) else None
    if not isinstance(task_id, str) or not task_id:
        created = _json_request("POST", _url(os.environ.get("RECA_WAN30_SUBMIT_PATH", _CREATE_PATH)), payload=payload)
        task_id = _task_id(created)
        state = {"request_id": request_id, "task_id": task_id, "model": model, "created": created}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    final = _poll(task_id)
    if _status(final) not in {"succeeded", "success", "completed"}:
        raise RuntimeError(f"Wan 3.0 task {task_id} ended with status={_status(final)!r}: {final}")
    video_url = _video_url(final)
    if not video_url:
        raise RuntimeError(f"Wan 3.0 task {task_id} succeeded but returned no video URL: {final}")
    _download(video_url, output_path)
    state.update({"status": "succeeded", "final": final, "provider_video_url": video_url})
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return video_url


class Wan30VideoBackend(VideoSegmentBackendBase):
    PROVIDER_SPEC = ProviderSpec(
        name="wan3.0-video",
        family="wan-video",
        supports_resolutions=("1280x720", "1920x1080"),
        segment_duration_range=(2.0, 15.0),
        bridge_duration_range=(2.0, 15.0),
        max_reference_images=5,
        max_prompt_chars=20000,
        concurrency_env="RECA_WAN30_WORKERS",
        default_concurrency=8,
        per_key_concurrency=5,
        cost_per_call_usd=0.0,
        provider="dashscope",
        supports_i2v=True,
        supports_r2v=True,
    )

    def render_segment(self, req: SegmentRequest) -> VideoResult:
        # The deployed ``wan3.0-video`` contract rejects a first_frame mixed
        # with reference_image (it only permits first_frame + last_frame for
        # bridge interpolation). Keep ReCA's continuity anchor as the hard
        # condition and omit soft refs for normal segments; bridge requests
        # below still use the provider-supported first/last pair.
        refs: list[str] = []
        media = [{"type": "first_frame", "url": req.first_url}]
        if (
            req.mode == "r2v"
            and os.environ.get("RECA_WAN30_ALLOW_MIXED_MEDIA", "0").strip().lower()
            in {"1", "true", "yes"}
        ):
            refs = [url for url in req.reference_image_urls if url][:5]
            media.extend({"type": "reference_image", "url": url} for url in refs)
        try:
            provider_url = _generate(
                request_id=req.request_id, prompt=req.prompt, media=media,
                duration_s=req.duration_s, seed=req.seed, output_path=req.output_path,
                log_dir=req.log_dir, resolution=req.resolution, negative_prompt=req.negative_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendRenderError(f"{self.PROVIDER_SPEC.name}({req.request_id}): {type(exc).__name__}: {str(exc)[:500]}") from exc
        return VideoResult(request_id=req.request_id, success=True, output_url=provider_url, output_path=req.output_path, rendered_duration_s=float(req.duration_s), seed_used=req.seed, backend_name=self.PROVIDER_SPEC.name)

    def render_bridge(self, req: BridgeRequest) -> VideoResult:
        media = [
            {"type": "first_frame", "url": req.first_url},
            {"type": "last_frame", "url": req.last_url},
        ]
        try:
            provider_url = _generate(
                request_id=req.request_id, prompt=req.prompt, media=media,
                duration_s=req.duration_s, seed=req.seed, output_path=req.output_path,
                log_dir=req.log_dir, resolution=req.resolution, negative_prompt=req.negative_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            raise BackendRenderError(f"{self.PROVIDER_SPEC.name}({req.request_id}): {type(exc).__name__}: {str(exc)[:500]}") from exc
        return VideoResult(request_id=req.request_id, success=True, output_url=provider_url, output_path=req.output_path, rendered_duration_s=float(req.duration_s), seed_used=req.seed, backend_name=self.PROVIDER_SPEC.name)


register_backend(Wan30VideoBackend.PROVIDER_SPEC.name, Wan30VideoBackend())


__all__ = ["Wan30VideoBackend"]
