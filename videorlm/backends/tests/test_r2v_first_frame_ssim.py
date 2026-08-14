"""E2E SSIM probe: does the rendered video's frame 0 actually match `first_url`?

Verifies what FLF2V backends ACTUALLY do on the wire vs what they claim
via ``caps.flf2v_mode``. Only runs when ``RECA_E2E_LIVE=1`` (otherwise
skipped — pytest collects but no-op so CI never spends API money).

Backends covered:
  - ``wan2.7-r2v``       (caps.flf2v_mode = "last_soft" / has first_frame slot)
  - ``happyhorse-1.0-r2v`` (caps.flf2v_mode = "all_soft" / no first_frame slot,
                            relies on prompt-prefix soft anchor)

Run::

    RECA_E2E_LIVE=1 python3 -m videorlm.backends.tests.test_r2v_first_frame_ssim
    # or
    RECA_E2E_LIVE=1 pytest videorlm/backends/tests/test_r2v_first_frame_ssim.py -s

Reads OSS / DashScope keys via the same auto_load_project_env path the rest
of the package uses. Each call is ~90s, so a full run is 3-4 minutes.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np


# Public OSS-hosted test image (re-uses earlier gpt-image-2 smoke test).
TEST_FIRST_URL = (
    "https://intern-data-wlcb.oss-cn-wulanchabu.aliyuncs.com/"
    "gpt-image-2/anchor_image/20260430/oss_e2e_test.png"
)

# Soft thresholds calibrated by hand from the 2026-04-30 SSIM probe and
# repo doc claims. Treat them as floors — actual SSIM should be much higher.
_SSIM_FLOOR_HARD = 0.55     # wan2.7-r2v with first_frame slot
_SSIM_FLOOR_SOFT = 0.20     # happyhorse-1.0-r2v with prompt-prefix only


_LIVE = os.environ.get("RECA_E2E_LIVE", "").strip() in ("1", "true", "True")


def _ensure_env() -> None:
    """No-op now — env reads go through ``backends._common.env.env_value``
    which already walks the repo's ``.env`` fallback chain on demand.
    Kept as a function so existing call sites stay path-stable."""
    return None


def _ssim_first_frame_vs_input(mp4_path: str, ref_url: str) -> float:
    """Decode mp4 frame 0, fetch ref_url, return grayscale SSIM in [-1, 1]."""
    import cv2
    from skimage.metrics import structural_similarity

    cap = cv2.VideoCapture(mp4_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame 0 from {mp4_path}")

    raw = httpx.get(ref_url, timeout=120).content
    arr = np.frombuffer(raw, dtype=np.uint8)
    ref = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if ref is None:
        raise RuntimeError(f"could not decode reference image from {ref_url}")

    h, w = frame.shape[:2]
    ref_resized = cv2.resize(ref, (w, h))
    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g_ref = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(g_frame, g_ref))


def _render_r2v_and_score(
    backend_name: str,
    *,
    out_dir: Path,
    seed: int = 42,
) -> tuple[float, float]:
    """Dispatch one r2v call to ``backend_name`` with TEST_FIRST_URL as first
    anchor and a single soft reference (same URL). Returns (ssim, elapsed_s).
    """
    import os as _os
    from backends.media import SegmentRequest, dispatch_segment

    out_path = out_dir / f"first_frame_anchor_{backend_name.replace('.', '_')}.mp4"
    request = SegmentRequest(
        request_id=f"e2e_anchor_{backend_name}",
        # Generic prompt — doesn't depend on the test image content.
        prompt="The same scene, slow camera dolly forward, photoreal still life.",
        first_url=TEST_FIRST_URL,
        mode="r2v",
        reference_image_urls=(TEST_FIRST_URL,),
        duration_s=5.0,
        seed=seed,
        output_path=str(out_path),
        log_dir=str(out_dir / "logs"),
    )
    t0 = time.time()
    # Route to the explicit backend under test via env (no chain / fallback).
    _os.environ["RECA_RENDER_BACKEND_SEGMENT_R2V"] = backend_name
    result = dispatch_segment(request)
    elapsed = time.time() - t0
    assert result.success, f"{backend_name} render failed: {result.error}"
    assert out_path.exists(), f"{backend_name} no mp4 at {out_path}"
    ssim = _ssim_first_frame_vs_input(str(out_path), TEST_FIRST_URL)
    return ssim, elapsed


def _gate_or_skip():
    if not _LIVE:
        try:
            import pytest
            pytest.skip("set RECA_E2E_LIVE=1 to spend ~$ on this live API test")
        except ImportError:
            print("[skip] RECA_E2E_LIVE not set; pass RECA_E2E_LIVE=1 to run.")
            return False
    return True


def test_wan_r2v_hard_first_frame_anchor(tmp_path):
    """wan2.7-r2v has a real first_frame slot — frame 0 must match input
    pixel-for-pixel-ish (SSIM ≥ 0.55, in practice we see > 0.85)."""
    if not _gate_or_skip():
        return
    _ensure_env()
    ssim, elapsed = _render_r2v_and_score("wan2.7-r2v", out_dir=tmp_path)
    print(f"[wan2.7-r2v] SSIM(frame_0, input) = {ssim:.4f}  elapsed={elapsed:.1f}s")
    assert ssim >= _SSIM_FLOOR_HARD, (
        f"wan2.7-r2v should HARD-anchor first frame "
        f"(SSIM ≥ {_SSIM_FLOOR_HARD}); got {ssim:.4f}"
    )


def test_happyhorse_r2v_soft_first_frame_anchor(tmp_path):
    """happyhorse-1.0-r2v has NO first_frame slot. Our backend injects a
    Chinese prompt prefix telling the model that reference[0] is the start
    frame — best-effort soft anchor. We expect SSIM materially > random
    noise but well under the hard-anchor backends."""
    if not _gate_or_skip():
        return
    _ensure_env()
    ssim, elapsed = _render_r2v_and_score("happyhorse-1.0-r2v", out_dir=tmp_path)
    print(
        f"[happyhorse-1.0-r2v] SSIM(frame_0, input) = {ssim:.4f}  "
        f"elapsed={elapsed:.1f}s  (soft anchor via prompt prefix)"
    )
    assert ssim >= _SSIM_FLOOR_SOFT, (
        f"happyhorse-r2v with prompt-prefix soft anchor should at least "
        f"reach SSIM ≥ {_SSIM_FLOOR_SOFT}; got {ssim:.4f} — prompt prefix "
        f"may not be flowing through correctly."
    )


def main() -> int:
    """Standalone runner: prints a comparison table and exits non-zero if
    any backend falls under its floor."""
    if not _LIVE:
        print("RECA_E2E_LIVE not set; pass RECA_E2E_LIVE=1 to spend ~$ on the run.")
        return 0
    _ensure_env()
    out_dir = Path("/tmp/videorlm_r2v_anchor_e2e")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float, float, str]] = []

    cases = [
        ("wan2.7-r2v",         _SSIM_FLOOR_HARD, "hard first_frame slot"),
        ("happyhorse-1.0-r2v", _SSIM_FLOOR_SOFT, "no slot;prompt prefix soft"),
    ]
    failures = 0
    for backend_name, floor, note in cases:
        try:
            ssim, elapsed = _render_r2v_and_score(backend_name, out_dir=out_dir)
            verdict = "PASS" if ssim >= floor else "FAIL"
            if ssim < floor:
                failures += 1
            rows.append((backend_name, ssim, floor, elapsed, f"{verdict}  ({note})"))
        except Exception as e:  # noqa: BLE001
            rows.append((backend_name, float("nan"), floor, 0.0,
                         f"ERROR  {type(e).__name__}: {str(e)[:80]}  ({note})"))
            failures += 1

    print()
    print(f"{'backend':24s} {'SSIM(frame_0,input)':>20s} {'floor':>8s} "
          f"{'elapsed':>10s}   verdict / note")
    print("-" * 96)
    for name, ssim, floor, elapsed, verdict in rows:
        ssim_str = f"{ssim:.4f}" if not np.isnan(ssim) else " nan "
        print(f"{name:24s} {ssim_str:>20s} {floor:>8.2f} {elapsed:>9.1f}s   {verdict}")
    print()
    print(f"output dir: {out_dir}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
