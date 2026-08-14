"""Minimal OSS publish helper for backend-side outputs.

Used by backends that produce binary content (b64 / local file) but need
to expose a public URL so downstream DashScope / wan calls can fetch it
as a reference. Currently:
  - ``gpt-image-2`` returns base64 → save locally → upload here →
    propagate as ``RenderResult.output_url``.

OSS config is read lazily from process env plus repo ``.env`` fallbacks:

  OSS_ACCESS_KEY_ID
  OSS_ACCESS_KEY_SECRET
  oss_AccessKey_ID / oss_AccessKey_Secret  (legacy repo-root aliases)
  oss_bucket / OSS_BUCKET / os_path / OSS_PATH (last two: ``oss://bucket/prefix``)
  region / OSS_REGION  (e.g. ``cn-hangzhou``) — endpoint inferred if not given
  OSS_ENDPOINT         (overrides region inference)

If any required field is missing, ``upload_*`` returns ``None`` (caller
falls back to local-only ``output_path`` and skips ``output_url``).
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from .env import env_value
from .trace import trace_event


def _infer_endpoint(region: str | None) -> str:
    if not region:
        return ""
    region = region.strip()
    if not region:
        return ""
    cn_map = {
        "华东1(杭州)": "cn-hangzhou",
        "华东2(上海)": "cn-shanghai",
        "华北1(青岛)": "cn-qingdao",
        "华北2(北京)": "cn-beijing",
        "华北3(张家口)": "cn-zhangjiakou",
        "华北5(呼和浩特)": "cn-huhehaote",
        "华北6(乌兰察布)": "cn-wulanchabu",
        "乌兰察布": "cn-wulanchabu",
        "华南1(深圳)": "cn-shenzhen",
        "华南2(河源)": "cn-heyuan",
        "华南3(广州)": "cn-guangzhou",
        "中国香港": "cn-hongkong",
        "香港": "cn-hongkong",
        "华西1(成都)": "cn-chengdu",
    }
    region = cn_map.get(region, region)
    if region.startswith("oss-"):
        return f"https://{region}.aliyuncs.com"
    if region.startswith(("cn-", "ap-", "eu-", "us-")):
        return f"https://oss-{region}.aliyuncs.com"
    return ""


def _infer_bucket_and_prefix(raw: str | None) -> tuple[str, str]:
    if not raw:
        return "", ""
    raw = raw.strip()
    if not raw:
        return "", ""
    if raw.startswith("oss://"):
        raw = raw[len("oss://"):]
    body = raw.strip("/")
    if not body:
        return "", ""
    parts = body.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _load_oss_config() -> tuple[str, str, str, str, str] | None:
    """Returns (access_key_id, access_key_secret, bucket, endpoint, base_prefix) or None.

    All resolution is local — ``env_value()`` already walks the repo's
    ``.env`` fallback chain (see ``_common/env.py``), and
    ``_infer_endpoint`` / ``_infer_bucket_and_prefix`` above handle the
    region → endpoint and ``oss://bucket/prefix`` parsing in-tree.
    """
    access_id = env_value(
        "OSS_ACCESS_KEY_ID",
        "ALIBABA_CLOUD_ACCESS_KEY_ID",
        "oss_AccessKey_ID",
        "oss_access_key_id",
    ).strip()
    access_secret = env_value(
        "OSS_ACCESS_KEY_SECRET",
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
        "oss_AccessKey_Secret",
        "oss_access_key_secret",
    ).strip()
    if not (access_id and access_secret):
        return None

    raw_path = env_value("os_path", "OSS_PATH", "OSS_BUCKET").strip()
    path_bucket, path_prefix = _infer_bucket_and_prefix(raw_path)

    bucket = (
        env_value("oss_bucket", "OSS_BUCKET")
        or path_bucket
        or ""
    ).strip()
    if not bucket:
        return None

    region = env_value("region", "OSS_REGION").strip()
    endpoint = env_value("OSS_ENDPOINT").strip() or _infer_endpoint(region)
    if not endpoint:
        return None

    return (access_id, access_secret, bucket, endpoint, (path_prefix or "").strip("/"))


def upload_bytes(
    content: bytes,
    *,
    suffix: str = ".png",
    prefix: str = "",
    request_id: str = "",
    log_dir: str | None = None,
) -> str | None:
    """Upload ``content`` to OSS. Returns public URL or None if OSS not configured.

    Object key: ``<base_prefix>/<prefix>/YYYYMMDD/<request_id_or_uuid><suffix>``
    """
    cfg = _load_oss_config()
    if cfg is None:
        trace_event(
            "oss.upload", "skipped", log_dir=log_dir,
            request_id=request_id,
            reason=(
                "OSS env not configured "
                "(OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET / oss_bucket / region missing)"
            ),
        )
        return None
    access_id, access_secret, bucket_name, endpoint, base_prefix = cfg

    try:
        import oss2
    except ImportError:
        trace_event(
            "oss.upload", "skipped", log_dir=log_dir,
            request_id=request_id, reason="oss2 SDK not installed",
        )
        return None

    parts: list[str] = []
    if base_prefix:
        parts.append(base_prefix)
    if prefix:
        parts.append(prefix.strip("/"))
    parts.append(time.strftime("%Y%m%d"))
    name_id = (request_id or uuid.uuid4().hex[:12]).strip("/").replace("/", "_")
    parts.append(f"{name_id}{suffix}")
    object_key = "/".join(p for p in parts if p)

    try:
        auth = oss2.Auth(access_id, access_secret)
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        bucket.put_object(object_key, content)
        bucket.put_object_acl(object_key, oss2.OBJECT_ACL_PUBLIC_READ)
    except Exception as e:  # noqa: BLE001
        trace_event(
            "oss.upload", "error", log_dir=log_dir,
            request_id=request_id,
            error_type=type(e).__name__, error=str(e)[:200],
        )
        return None

    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    public_url = f"https://{bucket_name}.{host}/{object_key}"
    trace_event(
        "oss.upload", "success", log_dir=log_dir,
        request_id=request_id, object_key=object_key, url=public_url,
    )
    return public_url


def upload_file(
    path: str,
    *,
    prefix: str = "",
    request_id: str = "",
    log_dir: str | None = None,
) -> str | None:
    """Read local file and upload to OSS. Returns public URL or None."""
    p = Path(path)
    if not p.exists():
        trace_event(
            "oss.upload", "skipped", log_dir=log_dir,
            request_id=request_id, reason=f"local file missing: {path}",
        )
        return None
    suffix = p.suffix or ".bin"
    return upload_bytes(
        p.read_bytes(),
        suffix=suffix,
        prefix=prefix,
        request_id=request_id,
        log_dir=log_dir,
    )
