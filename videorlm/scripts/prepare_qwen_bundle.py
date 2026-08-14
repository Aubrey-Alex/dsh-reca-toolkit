"""Prepare the qwen-planner + wan-image-pro batch results as an OSS bundle.

Per ``outputs/rule.md`` + ``outputs/oss_rule.md``:
  - Bundle goes to ``outputs/new_videos/<experiment>/``.
  - Must include ``experiment.json`` with title / category / subcategory.
  - Must include ``index.html`` (local debug) + ``index.oss.html`` (public).
  - Uploads via ``experimental_baseline/tools/oss_preview.py`` to
    bucket ``intern-data-wlcb`` at ``new-videos/<experiment>/<ts>`` prefix.
  - Final ``oss_upload_result.json`` records signed URLs.

This script:
  1. Scans ``outputs/version_2.4/ex??_{wan,happyhorse}/run/`` for slots with
     ``final.mp4`` already produced.
  2. Backfills ``_backend_info.json`` for slots that don't have one (the
     in-flight procs that imported pipeline.py before the backend_info
     write was added).
  3. Builds the bundle layout: copies final.mp4 + a contact-sheet frame
     + skeleton.json + summary.json per slot.
  4. Writes ``experiment.json`` + ``index.html`` describing the
     wan-vs-happyhorse comparison.
  5. Uploads + writes ``index.oss.html`` + ``oss_upload_result.json``.

Usage:
  python3 videorlm/scripts/prepare_qwen_bundle.py [--upload] [--ex 01,07,..] [--experiment NAME]

  --upload     actually push to OSS (default: dry-run, only build local bundle)
  --ex         comma-separated example ids to include (default: all MISS-recovered)
  --experiment override experiment dir name; default is timestamp-based.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/mnt/workspace/akide/code/unirlm-02")
RUN_ROOT = REPO / "videorlm/outputs/version_2.4"
BUNDLE_PARENT = REPO / "outputs/new_videos"

# Default experiment name follows rule.md naming: date-version-stack
DEFAULT_EXPERIMENT = f"{time.strftime('%Y%m%d')}-version-2-4-batch-qwen-wan27pro-rerun"

# Backend-tag defaults that match run_batch_qwen.sh / run_ex01_qwen.sh.
# Used for backfilling slots whose summary.json doesn't carry backend_info.
BACKFILL_BACKEND_INFO = {
    "tag": "wan27-image-pro+qwen3.6-max-preview-planner",
    "planner_model": "qwen3.6-max-preview",
    "render": {
        "portrait":     "wan2.7-image-pro",
        "anchor_image": "wan2.7-image-pro",
        "image_edit":   "wan2.7-image-pro",
        "segment_r2v":  "(wan/happyhorse, see slot label)",
        "segment_i2v":  "(wan/happyhorse, see slot label)",
        "bridge":       "wan2.7-i2v",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true", help="actually push to OSS")
    p.add_argument("--ex", default="", help="comma-separated example ids (default: all MISS-recovered)")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="experiment directory name")
    p.add_argument("--target-slots", default="01,07,13,17,19,23,29,31,32,37,38,39,40,41,42,43,44",
                   help="comma-separated example ids known to be in the qwen rerun batch")
    return p.parse_args()


def backfill_backend_info(slot_dir: Path) -> None:
    info_path = slot_dir / "_backend_info.json"
    if info_path.exists():
        return
    info_path.write_text(
        json.dumps(BACKFILL_BACKEND_INFO, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  backfilled _backend_info.json in {slot_dir}", flush=True)


def make_contact_sheet(final_mp4: Path, out_jpg: Path) -> bool:
    """Best-effort: 3x3 contact sheet via ffmpeg. Returns True if succeeded."""
    if not final_mp4.exists():
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(final_mp4),
             "-vf", "fps=1/8,scale=320:-1,tile=3x3",
             "-frames:v", "1", str(out_jpg)],
            check=True, capture_output=True, timeout=120,
        )
        return out_jpg.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"  contact sheet skipped: {exc}", flush=True)
        return False


def collect_slot(slot_dir: Path, dest: Path) -> dict[str, Any]:
    """Copy public artifacts from slot_dir into dest. Returns slot manifest dict."""
    dest.mkdir(parents=True, exist_ok=True)
    final = slot_dir / "run" / "final.mp4"
    if not final.exists():
        return {"ok": False, "reason": "final.mp4 missing"}

    backfill_backend_info(slot_dir)

    # Copy final.mp4 + skeleton + summary + backend_info
    shutil.copy2(final, dest / "final.mp4")
    for name in ("skeleton.json", "_backend_info.json"):
        src = slot_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    sm = slot_dir / "summary.json"
    if sm.exists():
        shutil.copy2(sm, dest / "summary.json")

    # First-frame image for the preview card
    first_frame = dest / "first_frame.jpg"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(final), "-vframes", "1",
             "-q:v", "3", str(first_frame)],
            check=True, capture_output=True, timeout=60,
        )
    except Exception:
        pass

    # Contact sheet
    make_contact_sheet(final, dest / "contact.jpg")

    backend_info = {}
    info_path = dest / "_backend_info.json"
    if info_path.exists():
        try:
            backend_info = json.loads(info_path.read_text())
        except Exception:
            pass

    return {
        "ok": True,
        "final_mp4": "final.mp4",
        "first_frame": "first_frame.jpg",
        "contact": "contact.jpg" if (dest / "contact.jpg").exists() else None,
        "backend_info": backend_info,
        "duration_s": _probe_duration(final),
        "size_bytes": final.stat().st_size,
    }


def _probe_duration(mp4: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


def render_index_html(bundle_dir: Path, slots: dict[str, dict[str, Any]],
                      experiment_meta: dict[str, Any]) -> None:
    """Write a self-contained index.html with wan-vs-happyhorse cards per ex."""
    backend_tag = experiment_meta.get("backend_tag", "wan27-image-pro+qwen3.6-max-preview")
    rows: list[str] = []
    by_ex: dict[str, dict[str, dict[str, Any]]] = {}
    for slot_id, info in slots.items():
        if not info.get("ok"):
            continue
        # slot_id like "ex07_wan"
        ex, _, backend = slot_id.partition("_")
        by_ex.setdefault(ex, {})[backend] = info

    for ex in sorted(by_ex.keys()):
        backends = by_ex[ex]
        cards = []
        for backend in ("wan", "happyhorse"):
            d = backends.get(backend)
            if not d:
                cards.append(f"<div class='card'><h4>{ex}_{backend} (missing)</h4></div>")
                continue
            dur = d.get("duration_s") or 0.0
            size_mb = (d.get("size_bytes") or 0) / 1024 / 1024
            backend_label = (d.get("backend_info") or {}).get("tag", "?")
            cards.append(f"""
<div class='card'>
  <h4>{ex}_{backend} <small>· {dur:.1f}s · {size_mb:.1f}MB · {backend_label}</small></h4>
  <video controls preload='metadata' src='{ex}_{backend}/final.mp4'></video>
  <details><summary>first frame</summary>
    <img src='{ex}_{backend}/first_frame.jpg' alt='first frame'/>
  </details>
  <details><summary>contact sheet</summary>
    <img src='{ex}_{backend}/contact.jpg' alt='contact sheet'/>
  </details>
</div>
""")
        rows.append(f"<section class='row'><h3>{ex}</h3>{''.join(cards)}</section>")

    body = "".join(rows) or "<p>No completed slots yet.</p>"
    html = f"""<!doctype html>
<html lang='zh-cn'>
<head>
<meta charset='utf-8'/>
<title>{experiment_meta.get('title', 'qwen-batch')}</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:24px;background:#0d1117;color:#e6edf3;}}
.row{{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:32px;padding-bottom:16px;border-bottom:1px solid #30363d;}}
.row h3{{flex-basis:100%;margin:0 0 8px 0;color:#7ee787;}}
.card{{flex:1 1 480px;max-width:640px;background:#161b22;padding:12px;border-radius:8px;}}
.card h4{{margin:0 0 8px 0;color:#79c0ff;font-weight:500;}}
.card small{{color:#8b949e;font-weight:400;}}
.card video{{width:100%;border-radius:4px;background:#000;}}
.card img{{width:100%;border-radius:4px;margin-top:8px;}}
.card details summary{{cursor:pointer;color:#8b949e;margin-top:8px;}}
.banner{{background:#1f2937;padding:16px;border-radius:8px;margin-bottom:24px;}}
a.back{{color:#79c0ff;}}
</style>
</head>
<body>
<a class='back' href='https://preview-videorlm.coolify-master.fsn1-de-sec.vmv.re/'>← 返回总入口</a>
<h1>{experiment_meta.get('title','qwen-batch rerun')}</h1>
<div class='banner'>
  <b>Backend stack:</b> {backend_tag}<br/>
  <b>Why this run exists:</b> {experiment_meta.get('notes','')}<br/>
  <b>Generated:</b> {experiment_meta.get('generated_at','')}
</div>
{body}
</body>
</html>
"""
    (bundle_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"[index] wrote {bundle_dir/'index.html'}", flush=True)


def upload_to_oss(bundle_dir: Path, experiment: str) -> dict[str, str]:
    """Upload bundle via oss_preview helper. Returns relpath → signed url map."""
    mod_path = REPO / "experimental_baseline/tools/oss_preview.py"
    spec = importlib.util.spec_from_file_location("oss_preview", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"oss_preview spec failed: {mod_path}")
    oss_preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oss_preview)  # type: ignore[attr-defined]

    bucket_name = "intern-data-wlcb"
    endpoint = "https://oss-cn-wulanchabu.aliyuncs.com"
    prefix = f"new-videos/{experiment}/{time.strftime('%Y%m%d-%H%M%S')}"
    expires_s = 30 * 24 * 3600

    oss_preview.auto_load_project_env()
    bucket = oss_preview.make_bucket(endpoint, bucket_name)
    uploaded = oss_preview.upload_bundle(bucket, bundle_dir, prefix)
    signed = oss_preview.build_signed_urls(bucket, uploaded, expires_s=expires_s)
    print(f"[oss] {len(uploaded)} files at prefix={prefix}", flush=True)

    # Rewrite index.html → index.oss.html with signed URLs
    html = (bundle_dir / "index.html").read_text(encoding="utf-8")
    for rel in sorted(signed, key=len, reverse=True):
        if rel in {"index.html", "index.oss.html"}:
            continue
        html = html.replace(f'src="{rel}"', f'src="{signed[rel]}"')
        html = html.replace(f'href="{rel}"', f'href="{signed[rel]}"')
    (bundle_dir / "index.oss.html").write_text(html, encoding="utf-8")

    # Re-upload index.oss.html
    oss_preview.upload_bundle(bucket, bundle_dir, prefix)
    signed = oss_preview.build_signed_urls(bucket, uploaded, expires_s=expires_s)

    (bundle_dir / "oss_upload_result.json").write_text(
        json.dumps({
            "bucket": bucket_name,
            "endpoint": endpoint,
            "prefix": prefix,
            "expires_s": expires_s,
            "signed_urls": signed,
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return signed


def main() -> int:
    args = parse_args()
    targets = [s.strip() for s in (args.ex or args.target_slots).split(",") if s.strip()]
    bundle = BUNDLE_PARENT / args.experiment
    bundle.mkdir(parents=True, exist_ok=True)

    slots: dict[str, dict[str, Any]] = {}
    for ex in targets:
        for backend in ("wan", "happyhorse"):
            slot_id = f"ex{ex}_{backend}"
            slot_dir = RUN_ROOT / slot_id
            if not slot_dir.exists():
                continue
            info = collect_slot(slot_dir, bundle / slot_id)
            slots[slot_id] = info
            status = "OK" if info.get("ok") else f"SKIP({info.get('reason','?')})"
            print(f"[slot] {slot_id}: {status}", flush=True)

    experiment_meta = {
        "title": "ex01-ex44 MISS rerun · qwen3.6-max-preview planner + wan2.7-image-pro renders",
        "category": "Multi-shot Narrative",
        "subcategory": "wan vs happyhorse · qwen-image fallback stack",
        "backend_tag": "wan27-image-pro+qwen3.6-max-preview-planner",
        "notes": (
            "Recovery rerun for slots that failed the gpt-5.5/gpt-image-2 stack "
            "due to gateway/CF 524 long-tail (verified 2026-05-15). Planner LLM "
            "switched to qwen3.6-max-preview on DashScope; image-gen kinds "
            "(portrait/anchor/edit) switched to wan2.7-image-pro on DashScope; "
            "everything else identical to the 2026-05-14 batch_18."
        ),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slots_total": sum(1 for v in slots.values() if v.get("ok")),
    }
    (bundle / "experiment.json").write_text(
        json.dumps(experiment_meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    render_index_html(bundle, slots, experiment_meta)

    if args.upload:
        try:
            signed = upload_to_oss(bundle, args.experiment)
            print(f"[oss] index.oss.html: {signed.get('index.oss.html','?')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[oss] FAIL: {type(exc).__name__}: {exc}", flush=True)
            return 1
    else:
        print(f"[oss] dry run; pass --upload to push. Bundle ready at {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
