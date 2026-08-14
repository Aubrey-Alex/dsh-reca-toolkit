"""LTX-2.3 inference runner — drives the LTX-2 pipelines via subprocess.

The subprocess approach is deliberate (see ``backend.py`` docstring):
``_runner`` MUST NOT import ``torch`` / ``diffusers`` / ``transformers``
into the videorlm main process. Instead each render shells out to the
LTX-2 pipeline CLI inside its own Python environment.

Env knobs (override with shell exports or ``.env``):

  Minimum (the only one most users need to set):
  - ``LTX23_ROOT``              — directory containing the LTX-2.3 weight
                                  files. ``checkpoint``,
                                  ``distilled_lora``, and
                                  ``spatial_upsampler`` are resolved from
                                  it by filename.

  Optional overrides:
  - ``LTX23_PYTHON``            — interpreter that has ``ltx_pipelines``
                                  importable (default: conda ``ltx2``).
  - ``LTX23_GEMMA_ROOT``        — Gemma 3 text-encoder snapshot dir.
  - ``LTX23_GPU``               — GPU index for ``CUDA_VISIBLE_DEVICES``
                                  (default ``0``).
  - ``LTX23_DISTILLED_LORA_STRENGTH`` — default ``0.8``.
  - ``LTX23_TIMEOUT_S``         — render timeout (default 3600).
  - ``LTX23_FRAME_RATE``        — default ``24``.

  Per-file overrides (rarely needed; use ``LTX23_ROOT`` instead):
  - ``LTX23_CHECKPOINT`` / ``LTX23_DISTILLED_LORA`` /
    ``LTX23_SPATIAL_UPSAMPLER``.

Resolution handling
-------------------

LTX-2.3 demands ``height % 64 == 0 and width % 64 == 0`` for stage-2 output
(stage-1 is half that, ``% 32``). videorlm conventionally requests
``1280x720`` / ``1920x1080`` which are NOT 64-aligned on height. We:

  1. Round to the nearest 64-multiple INSIDE LTX (e.g. ``720 → 704``,
     ``1080 → 1088``).
  2. After LTX produces the mp4 at the aligned resolution, ffmpeg
     ``pad`` / ``crop`` it back to the caller's requested resolution so
     downstream concat operations get a clean match.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────

_LTX_SNAPSHOT_DEFAULT = (
    "/mnt/workspace/akide/models/huggingface/hub/"
    "models--Lightricks--LTX-2.3/snapshots/"
    "76730e634e70a28f4e8d51f5e29c08e40e2d8e74"
)
_GEMMA_SNAPSHOT_DEFAULT = (
    "/mnt/workspace/akide/models/huggingface/hub/"
    "models--Lightricks--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/"
    "d62fe4f1995ade703b49a0f3c0d0f161237ef437"
)
_CHECKPOINT_FILENAME = "ltx-2.3-22b-dev.safetensors"
_DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
_SPATIAL_UPSAMPLER_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

_SIMPLE_DEFAULTS = {
    "python": "/mnt/workspace/akide/anaconda3/envs/ltx2/bin/python",
    "distilled_lora_strength": "0.8",
    "gemma_root": _GEMMA_SNAPSHOT_DEFAULT,
    "gpu": "0",
    "timeout_s": "3600",
    "frame_rate": "24",
}


def _ltx_root() -> str:
    return os.environ.get("LTX23_ROOT", _LTX_SNAPSHOT_DEFAULT)


def _cfg(name: str) -> str:
    """Resolve an LTX runner config field from env or default.

    ``checkpoint`` / ``distilled_lora`` / ``spatial_upsampler`` derive
    from ``LTX23_ROOT`` unless their own env override is set.
    """
    env_key = f"LTX23_{name.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    if name == "checkpoint":
        return f"{_ltx_root()}/{_CHECKPOINT_FILENAME}"
    if name == "distilled_lora":
        return f"{_ltx_root()}/{_DISTILLED_LORA_FILENAME}"
    if name == "spatial_upsampler":
        return f"{_ltx_root()}/{_SPATIAL_UPSAMPLER_FILENAME}"
    return _SIMPLE_DEFAULTS[name]


def _validate_assets() -> None:
    """Fail loudly if any required weight/path is missing."""
    missing: list[str] = []
    for key in ("python", "checkpoint", "distilled_lora", "spatial_upsampler", "gemma_root"):
        path = _cfg(key)
        if not Path(path).exists():
            missing.append(f"LTX23_{key.upper()}={path}")
    if missing:
        raise FileNotFoundError(
            "LTX-2.3 runner: required asset(s) missing on disk:\n  - "
            + "\n  - ".join(missing)
        )


# ── URL / path resolution ───────────────────────────────────────────────────


def _resolve_to_local(url: str, *, scratch: Path) -> str:
    """Return a local file path for ``url``.

    - ``file://...`` and bare absolute paths return as-is.
    - ``http(s)://...`` is downloaded into ``scratch``.
    """
    if not url:
        raise ValueError("empty url")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else url
        if not Path(path).exists():
            raise FileNotFoundError(f"LTX-2.3 runner: local path not found: {path}")
        return path
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"LTX-2.3 runner: unsupported url scheme: {url}")
    suffix = Path(parsed.path).suffix or ".jpg"
    target = scratch / f"in_{abs(hash(url))}{suffix}"
    with urllib.request.urlopen(url, timeout=120) as resp, target.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return str(target)


# ── Resolution / frame-count helpers ────────────────────────────────────────


def _parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse ``"1920x1080"`` → ``(1920, 1080)``. Strict — raises ValueError."""
    if not resolution or "x" not in resolution:
        raise ValueError(f"bad resolution: {resolution!r}")
    w_s, h_s = resolution.lower().split("x", 1)
    return int(w_s), int(h_s)


def _round_to_multiple(v: int, m: int) -> int:
    """Round-half-up to nearest multiple of ``m``, minimum ``m``."""
    return max(m, ((v + m // 2) // m) * m)


def _ltx_render_dims(requested_w: int, requested_h: int) -> tuple[int, int]:
    """LTX-2 stage-2 output must be a multiple of 64 on both axes."""
    return _round_to_multiple(requested_w, 64), _round_to_multiple(requested_h, 64)


def _frames_for_duration(duration_s: float, frame_rate: float) -> int:
    """LTX-2 num_frames must satisfy ``frames = 8*k + 1``."""
    raw = max(1, int(round(duration_s * frame_rate)))
    k = max(1, round((raw - 1) / 8))
    return 8 * k + 1


# ── ffmpeg post-process ─────────────────────────────────────────────────────


def _ffmpeg_resize(src: Path, dst: Path, target_w: int, target_h: int) -> None:
    """Force ``src`` to exactly ``target_w x target_h`` and write to ``dst``.

    Uses ``scale + pad/crop`` to preserve aspect when possible. We always
    re-encode with libx264 so downstream concat sees a uniform stream.
    """
    if dst.exists():
        dst.unlink()
    vf = (
        f"scale='if(gt(a,{target_w}/{target_h}),{target_w},-2)':"
        f"'if(gt(a,{target_w}/{target_h}),-2,{target_h})',"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
        f"crop={target_w}:{target_h}"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg resize failed (rc={result.returncode}):\n"
            f"  stderr: {result.stderr[-500:]}"
        )


# ── Subprocess driver ───────────────────────────────────────────────────────


def _run_ltx_subprocess(
    *,
    pipeline_module: str,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
    images: list[tuple[str, int, float, int]],
    log_dir: str | None,
    request_id: str,
) -> None:
    """Launch the LTX-2 pipeline CLI and write to ``output_path``."""
    _validate_assets()
    cmd: list[str] = [
        _cfg("python"), "-m", pipeline_module,
        "--checkpoint-path", _cfg("checkpoint"),
        "--distilled-lora", _cfg("distilled_lora"), _cfg("distilled_lora_strength"),
        "--spatial-upsampler-path", _cfg("spatial_upsampler"),
        "--gemma-root", _cfg("gemma_root"),
        "--prompt", prompt,
        "--num-frames", str(num_frames),
        "--frame-rate", str(frame_rate),
        "--height", str(height),
        "--width", str(width),
        "--seed", str(seed),
        "--output-path", str(output_path),
    ]
    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    for path, frame_idx, strength, crf in images:
        cmd.extend(["--image", path, str(frame_idx), f"{strength:.3f}", str(crf)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = _cfg("gpu")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Persist HF cache pointer so the subprocess finds the local snapshots.
    env.setdefault("HF_HOME", "/mnt/workspace/akide/models/huggingface")
    env.setdefault("HF_HUB_CACHE", "/mnt/workspace/akide/models/huggingface/hub")

    log_path: Path | None = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"ltx_{request_id}.log"

    timeout = float(_cfg("timeout_s"))
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ) as proc:
        captured: list[str] = []
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                captured.append(line)
                if log_path is not None:
                    with log_path.open("a") as f:
                        f.write(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(
                f"LTX-2.3 {pipeline_module} timed out after {timeout}s "
                f"(request_id={request_id})"
            )

    if proc.returncode != 0:
        tail = "".join(captured[-30:])
        raise RuntimeError(
            f"LTX-2.3 {pipeline_module} exit={proc.returncode} "
            f"(request_id={request_id})\n  cmd: {shlex.join(cmd)}\n"
            f"  tail:\n{tail}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"LTX-2.3 {pipeline_module} returned 0 but no output file at "
            f"{output_path} (request_id={request_id})"
        )


# ── Public entry points ─────────────────────────────────────────────────────


def _normalize_request_paths(output_path: str, request_id: str) -> Path:
    if not output_path:
        raise ValueError(
            f"LTX-2.3 runner({request_id}): output_path is required"
        )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def run_segment(
    *,
    request_id: str,
    prompt: str,
    first_url: str,
    reference_image_urls: tuple[str, ...] = (),
    duration_s: float = 5.0,
    seed: int = 0,
    output_path: str = "",
    resolution: str = "1920x1080",
    log_dir: str | None = None,
    mode: str = "i2v",
    negative_prompt: str = "",
) -> str:
    """Render a single segment: first frame anchor + optional reference set.

    For both ``i2v`` (first only) and ``r2v`` (first + refs) we route through
    ``ti2vid_two_stages``. ``r2v`` mode currently maps reference images to
    successive keyframe positions; for finer-grained reference control, the
    caller should switch the backend env to an IC-LoRA pipeline (TODO).
    """
    final_out = _normalize_request_paths(output_path, request_id)
    requested_w, requested_h = _parse_resolution(resolution)
    ltx_w, ltx_h = _ltx_render_dims(requested_w, requested_h)
    frame_rate = float(_cfg("frame_rate"))
    num_frames = _frames_for_duration(duration_s, frame_rate)

    with tempfile.TemporaryDirectory(prefix=f"ltx23_seg_{request_id}_") as tmp:
        scratch = Path(tmp)
        first_path = _resolve_to_local(first_url, scratch=scratch)
        images: list[tuple[str, int, float, int]] = [(first_path, 0, 1.0, 33)]
        if mode == "r2v" and reference_image_urls:
            # Drop refs onto distinct frame_idx beyond 0 — they become weak
            # keyframe-style hints. videorlm typically passes ≤3 refs.
            for idx, ref_url in enumerate(reference_image_urls, start=1):
                if idx >= num_frames:
                    break
                ref_path = _resolve_to_local(ref_url, scratch=scratch)
                # Strength 0.5 so refs don't override motion entirely.
                images.append((ref_path, idx * (num_frames // (len(reference_image_urls) + 1)), 0.5, 33))

        ltx_out = scratch / "ltx_out.mp4"
        _run_ltx_subprocess(
            pipeline_module="ltx_pipelines.ti2vid_two_stages",
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=ltx_out,
            seed=seed,
            width=ltx_w,
            height=ltx_h,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            log_dir=log_dir,
            request_id=request_id,
        )
        if (ltx_w, ltx_h) != (requested_w, requested_h):
            _ffmpeg_resize(ltx_out, final_out, requested_w, requested_h)
        else:
            shutil.copy(ltx_out, final_out)

    return str(final_out)


def run_bridge(
    *,
    request_id: str,
    prompt: str,
    first_url: str,
    last_url: str,
    duration_s: float = 5.0,
    seed: int = 0,
    output_path: str = "",
    resolution: str = "1920x1080",
    log_dir: str | None = None,
    negative_prompt: str = "",
) -> str:
    """Render a chunk-boundary bridge using keyframe interpolation."""
    final_out = _normalize_request_paths(output_path, request_id)
    requested_w, requested_h = _parse_resolution(resolution)
    ltx_w, ltx_h = _ltx_render_dims(requested_w, requested_h)
    frame_rate = float(_cfg("frame_rate"))
    num_frames = _frames_for_duration(duration_s, frame_rate)

    with tempfile.TemporaryDirectory(prefix=f"ltx23_brg_{request_id}_") as tmp:
        scratch = Path(tmp)
        first_path = _resolve_to_local(first_url, scratch=scratch)
        last_path = _resolve_to_local(last_url, scratch=scratch)
        images = [
            (first_path, 0, 1.0, 33),
            (last_path, num_frames - 1, 1.0, 33),
        ]

        ltx_out = scratch / "ltx_out.mp4"
        _run_ltx_subprocess(
            pipeline_module="ltx_pipelines.keyframe_interpolation",
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=ltx_out,
            seed=seed,
            width=ltx_w,
            height=ltx_h,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            log_dir=log_dir,
            request_id=request_id,
        )
        if (ltx_w, ltx_h) != (requested_w, requested_h):
            _ffmpeg_resize(ltx_out, final_out, requested_w, requested_h)
        else:
            shutil.copy(ltx_out, final_out)

    return str(final_out)


__all__ = ["run_segment", "run_bridge"]
