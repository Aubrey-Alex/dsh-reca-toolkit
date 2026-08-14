"""HappyHorse 1.0 family — shared submit / poll / download.

The dashscope SDK's ``VideoSynthesis`` helper does not cover the happyhorse
endpoints, so we drive them via raw HTTP. All four model codes (t2v / i2v
/ r2v / video-edit) share the same task-queue endpoint; only the payload
shape differs per kind.
"""
from __future__ import annotations

import os
import time
from http import HTTPStatus
from pathlib import Path

import httpx

from ....interface.capabilities import BackendRenderError
from ....._common.dashscope_sdk import classify_dashscope_response
from ....._common.platforms import with_key
from ....._common.trace import trace_event


_SUBMIT_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "video-generation/video-synthesis"
)
_TASK_URL_TPL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

_POLL_INTERVAL_S = float(os.environ.get("RECA_HAPPYHORSE_POLL_INTERVAL_S", "5"))
# happyhorse actual render is 2-6 min, but queue wait can balloon when many
# tasks are in flight. 1h covers worst case observed (70-min queue + 6-min
# render) without re-submitting and creating zombie tasks.
_POLL_MAX_WAIT_S = float(os.environ.get("RECA_HAPPYHORSE_POLL_MAX_WAIT_S", "3600"))


def resolution_param(resolution: str | None, default: str = "720P") -> str:
    """DashScope HappyHorse expects ``720P`` / ``1080P`` not ``1280x720``."""
    if not resolution:
        return default
    if "x" in resolution:
        mapping = {"1280x720": "720P", "1920x1080": "1080P"}
        return mapping.get(resolution, default)
    return resolution


def submit_task(
    payload: dict, *, model_name: str, request_id: str,
    log_dir: str | None = None,
) -> str:
    def _do_submit(api_key: str) -> httpx.Response:
        return httpx.post(
            _SUBMIT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json=payload,
            timeout=120,
        )

    try:
        rsp = with_key(
            "dashscope",
            _do_submit,
            classify_response=classify_dashscope_response,
        )
    except Exception as exc:  # noqa: BLE001
        raise BackendRenderError(
            f"{model_name}.submit({request_id}): "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    if rsp.status_code != HTTPStatus.OK:
        raise BackendRenderError(
            f"{model_name}.submit({request_id}): "
            f"http={rsp.status_code} body={rsp.text[:200]}"
        )
    try:
        data = rsp.json()
    except Exception:  # noqa: BLE001
        data = {}
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        raise BackendRenderError(
            f"{model_name}.submit({request_id}): "
            f"missing task_id in response: {str(data)[:200]}"
        )
    # Emit task_id into trace_log so frontend can enumerate / cancel
    # DashScope zombie tasks when client is killed.
    trace_event(
        "video.happyhorse", "task_submitted",
        log_dir=log_dir,
        backend=model_name, request_id=request_id, task_id=task_id,
    )
    return task_id


def poll_until_ready(
    task_id: str, *, model_name: str, request_id: str,
    log_dir: str | None = None,
) -> str:
    url = _TASK_URL_TPL.format(task_id=task_id)
    deadline = time.time() + _POLL_MAX_WAIT_S
    last_status = ""

    def _do_poll(api_key: str) -> httpx.Response:
        return httpx.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
        )

    while time.time() < deadline:
        time.sleep(_POLL_INTERVAL_S)
        try:
            rsp = with_key(
                "dashscope",
                _do_poll,
                classify_response=classify_dashscope_response,
            )
        except Exception:  # noqa: BLE001
            continue
        if rsp.status_code != HTTPStatus.OK:
            continue
        try:
            data = rsp.json()
        except Exception:  # noqa: BLE001
            continue
        output = data.get("output") or {}
        last_status = str(output.get("task_status", "")).upper()
        if last_status == "SUCCEEDED":
            video_url = output.get("video_url") or ""
            if not video_url:
                raise BackendRenderError(
                    f"{model_name}.poll({request_id}): SUCCEEDED but no video_url"
                )
            trace_event(
                "video.happyhorse", "task_done",
                log_dir=log_dir,
                backend=model_name, request_id=request_id, task_id=task_id,
                status="SUCCEEDED",
            )
            return video_url
        if last_status == "FAILED":
            msg = output.get("message") or data.get("message") or ""
            trace_event(
                "video.happyhorse", "task_done",
                log_dir=log_dir,
                backend=model_name, request_id=request_id, task_id=task_id,
                status="FAILED", error=str(msg)[:200],
            )
            raise BackendRenderError(
                f"{model_name}.poll({request_id}): FAILED — {str(msg)[:200]}"
            )

    trace_event(
        "video.happyhorse", "task_done",
        log_dir=log_dir,
        backend=model_name, request_id=request_id, task_id=task_id,
        status="TIMEOUT", last_status=last_status,
    )
    raise BackendRenderError(
        f"{model_name}.poll({request_id}): timed out after "
        f"{_POLL_MAX_WAIT_S}s (last_status={last_status!r})"
    )


def download(video_url: str, output_path: str, *, model_name: str, request_id: str) -> None:
    try:
        mp4 = httpx.get(video_url, timeout=300).content
    except Exception as exc:  # noqa: BLE001
        raise BackendRenderError(
            f"{model_name}.download({request_id}): "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(mp4)


def cancel_task(task_id: str, *, model_name: str = "happyhorse") -> bool:
    """Best-effort cancel a happyhorse task. Returns True on HTTP 200.

    Tries every configured DashScope key — task ownership is per-key so a
    cancel only succeeds on the key that submitted. Used when client gives
    up (TIMEOUT exhausted, process kill) to free server queue slots.
    """
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}/cancel"
    keys_env = (
        os.environ.get("DASHSCOPE_API_KEYS")
        or os.environ.get("DASHSCOPE_API_KEY", "")
    )
    for k in (k.strip() for k in keys_env.split(",") if k.strip()):
        try:
            r = httpx.post(
                url, headers={"Authorization": f"Bearer {k}"}, timeout=10,
            )
            if r.status_code == HTTPStatus.OK:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def submit_poll_download(
    payload: dict,
    output_path: str,
    *,
    model_name: str,
    request_id: str,
    log_dir: str | None = None,
) -> str:
    """Run submit → poll → download with task_id persistence.

    A ``<output_path>.task_id`` sidecar is written immediately after submit
    and reused across retries / ``--resume`` so we never accidentally submit
    duplicate renders of the same segment (which used to flood the dashscope
    queue and starve every other task).

    State machine:
      * sidecar exists → poll that task_id; never re-submit
      * poll succeeds  → download, delete sidecar
      * poll FAILED    → delete sidecar (next retry submits fresh)
      * poll TIMEOUT   → keep sidecar (next retry continues polling)
      * non-recoverable error → cancel task + delete sidecar
    """
    if not output_path:
        raise BackendRenderError(
            f"{model_name}.render({request_id}): output_path required"
        )
    task_id_file = Path(str(output_path) + ".task_id")
    if task_id_file.exists():
        task_id = task_id_file.read_text(encoding="utf-8").strip()
        trace_event(
            "video.happyhorse", "task_reused",
            log_dir=log_dir,
            backend=model_name, request_id=request_id, task_id=task_id,
        )
    else:
        task_id = submit_task(
            payload, model_name=model_name, request_id=request_id, log_dir=log_dir,
        )
        task_id_file.parent.mkdir(parents=True, exist_ok=True)
        task_id_file.write_text(task_id, encoding="utf-8")
    try:
        video_url = poll_until_ready(
            task_id, model_name=model_name, request_id=request_id, log_dir=log_dir,
        )
    except BackendRenderError as exc:
        msg = str(exc)
        if "FAILED" in msg or "no video_url" in msg:
            # Server rejected — task_id is dead, let next retry submit fresh
            task_id_file.unlink(missing_ok=True)
        # On TIMEOUT (last_status='PENDING'/'RUNNING'), KEEP the sidecar so
        # the next retry / --resume re-polls the same task instead of
        # spawning a duplicate.
        raise
    download(video_url, output_path, model_name=model_name, request_id=request_id)
    task_id_file.unlink(missing_ok=True)
    return video_url
