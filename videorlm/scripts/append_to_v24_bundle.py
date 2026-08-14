"""Merge qwen-rerun slots into the existing v2.4 bundle dir.

Strategy:
  - Do NOT regenerate the existing index.html (it's 184 KB of carefully
    laid-out sections per slot, with full prompt / anchor / segment grids).
  - Instead: for each new slot, copy its run/ artifacts to
    ``assets/<slot>/`` and APPEND a fresh ``<section class="run">`` block
    to the existing index.html, just before ``</main>``. Update the TOC
    + manifest.json + experiment.json.

  - Each appended section is tagged with a `<span class="badge hot">qwen
    rerun</span>` so it's visually distinct from the original gpt-5.5 /
    gpt-image-2 sections.

Slot-source classification (recorded in ``_backend_info.json`` per slot
when ``pipeline.py`` finishes; backfilled here for slots that completed
before that pipeline edit landed):

  pre-batch     gpt-5.5 + gpt-image-2     (the 32 successful ex02-19 slots)
  qwen-rerun    qwen3.6-max-preview +     (the 16 MISS slots being added now:
                wan2.7-image-pro          ex01_{wan,hh}, ex07/19/23/29/37-44
                                          _wan, ex13/17/31/32_happyhorse)

Usage:
  python3 videorlm/scripts/append_to_v24_bundle.py [--upload]

  --upload   actually push the updated bundle to OSS + regenerate index.oss.html
  --slots    comma-separated specific slot names (default: scan all version_2.4
             slots with final.mp4 that aren't in the bundle yet)
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

REPO = Path("/mnt/workspace/akide/code/unirlm-02")
RUN_ROOT = REPO / "videorlm/outputs/version_2.4"
BUNDLE = REPO / "outputs/new_videos/20260514-version-2-4-batch-18-wan-vs-happyhorse"

# All slots the qwen rerun is meant to recover (single source of truth so
# we don't accidentally re-pull a pre-batch slot that happened to already
# be in the bundle).
QWEN_RERUN_SLOTS = (
    "ex01_wan", "ex01_happyhorse",
    "ex07_wan",
    "ex13_happyhorse",
    "ex17_happyhorse",
    "ex19_wan",
    "ex23_wan",
    "ex29_wan",
    "ex31_happyhorse",
    "ex32_happyhorse",
    "ex37_wan", "ex38_wan", "ex39_wan", "ex40_wan",
    "ex41_wan", "ex42_wan", "ex43_wan", "ex44_wan",
)

# Backfill stack — slots that finished before the pipeline.py
# `_backend_info.json` write was added (Stop hook timeline 2026-05-15
# evening). New runs starting after that auto-write it.
BACKFILL_INFO = {
    "tag": "wan27-image-pro+qwen3.6-max-preview-planner",
    "planner_model": "qwen3.6-max-preview",
    "render": {
        "portrait":     "wan2.7-image-pro",
        "anchor_image": "wan2.7-image-pro",
        "image_edit":   "wan2.7-image-pro",
        "bridge":       "wan2.7-i2v",
    },
    "note": (
        "Recovered run: planner = qwen3.6-max-preview on DashScope (bypasses "
        "gateway/CF 524 long-tail that truncated the original gpt-5.5 planner "
        "output at ~16k chars); image-gen = wan2.7-image-pro on DashScope "
        "(bypasses OpenAI-compat gpt-image-2 same CF 524 issue)."
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    p.add_argument("--slots", default="", help="comma-separated slot names")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be done, write nothing")
    return p.parse_args()


def find_completed_new_slots(explicit: list[str]) -> list[str]:
    """Return slot names that have new final.mp4 and aren't already in bundle."""
    if explicit:
        candidates = explicit
    else:
        candidates = list(QWEN_RERUN_SLOTS)
    out: list[str] = []
    for slot in candidates:
        src = RUN_ROOT / slot / "run" / "final.mp4"
        in_bundle = (BUNDLE / "assets" / slot / "final.mp4")
        if not src.exists():
            print(f"  skip {slot}: no final.mp4 in source", flush=True)
            continue
        if in_bundle.exists() and in_bundle.stat().st_mtime >= src.stat().st_mtime:
            print(f"  skip {slot}: already in bundle and newer/equal", flush=True)
            continue
        out.append(slot)
    return out


def copy_slot_assets(slot: str, dry: bool) -> dict:
    """Copy slot artifacts into bundle/assets/<slot>/, return metadata dict."""
    src = RUN_ROOT / slot
    dst = BUNDLE / "assets" / slot
    if dry:
        print(f"  [dry] would copy {src}/run/ → {dst}/", flush=True)
        return {}
    dst.mkdir(parents=True, exist_ok=True)
    # Key assets only: final.mp4 + skeleton.json + summary.json + _backend_info.json
    # + first-frame poster image (extracted via ffmpeg)
    for name in ("skeleton.json", "summary.json"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    final_src = src / "run" / "final.mp4"
    final_dst = dst / "final.mp4"
    shutil.copy2(final_src, final_dst)

    # _backend_info.json: prefer the one pipeline.py wrote at finalization;
    # else backfill from BACKFILL_INFO so OSS consumers know stack used.
    info_src = src / "_backend_info.json"
    info_dst = dst / "_backend_info.json"
    if info_src.exists():
        shutil.copy2(info_src, info_dst)
    else:
        info_dst.write_text(
            json.dumps(BACKFILL_INFO, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    # Poster + contact sheet for the index page
    poster = dst / "contact.jpg"
    if not poster.exists():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(final_src),
                 "-vf", "fps=1/8,scale=320:-1,tile=3x3",
                 "-frames:v", "1", str(poster)],
                check=True, capture_output=True, timeout=120,
            )
        except Exception as exc:
            print(f"    contact sheet skipped ({exc})", flush=True)

    meta = {
        "size_bytes": final_dst.stat().st_size,
        "duration_s": _probe_duration(final_dst),
        "backend_info": json.loads(info_dst.read_text()),
    }
    return meta


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


def build_section_html(slot: str, meta: dict) -> str:
    """One <section class="run"> block mirroring the existing index.html style."""
    backend = "wan" if slot.endswith("_wan") else "happyhorse"
    res = "1280x720" if backend == "wan" else "1920x1080"
    resolution_badge = "720P" if backend == "wan" else "1080P"
    size_mb = (meta.get("size_bytes") or 0) / 1024 / 1024
    dur = meta.get("duration_s") or 0.0
    info = meta.get("backend_info") or BACKFILL_INFO
    tag = info.get("tag", "qwen rerun")
    planner = info.get("planner_model", "?")
    render = info.get("render", {}) or {}
    img_backend = render.get("portrait", "?")
    note = info.get("note", "")

    return f"""
<section class="run" id="{slot}">
  <div class="run-head">
    <div>
      <h2>{slot} <span class="badge hot">{resolution_badge}</span> <span class="badge hot" style="background:#7c3aed;color:#fff">qwen rerun</span></h2>
      <p>final.mp4 · {res} · {dur:.1f}s · {size_mb:.1f} MB</p>
      <p class="path">videorlm/outputs/version_2.4/{slot} · planner={planner} · image={img_backend}</p>
    </div>
    <a class="top" href="#top">Back to top</a>
  </div>
  <video class="main-video" controls preload="none" loading="lazy" poster="assets/{slot}/contact.jpg"><source src="assets/{slot}/final.mp4" type="video/mp4"></video>
  <div class="meta">
    <span>backend tag: {tag}</span>
    <span>planner: {planner}</span>
    <span>image-gen: {img_backend}</span>
  </div>
  <details>
    <summary>Recovery context</summary>
    <pre>{note}</pre>
  </details>
</section>
"""


def merge_into_index(new_slots: list[str], slot_meta: dict[str, dict],
                     dry: bool) -> None:
    """Append new <section class="run"> blocks just before </main>. Also
    update the .toc grid if present."""
    if dry:
        print(f"  [dry] would append {len(new_slots)} sections to index.html", flush=True)
        return

    index_html = BUNDLE / "index.html"
    html = index_html.read_text(encoding="utf-8")

    # Idempotency: drop any prior appended block with the same id before
    # writing the new one.
    for slot in new_slots:
        marker = f'id="{slot}"'
        if marker in html:
            # Crude removal: find the section and drop it.
            start = html.find(f'<section class="run" id="{slot}"')
            if start >= 0:
                end = html.find("</section>", start)
                if end >= 0:
                    html = html[:start] + html[end + len("</section>"):]

    sections = "".join(build_section_html(s, slot_meta[s]) for s in new_slots)
    # Add a banner above the new sections to distinguish them.
    banner = f"""
<section class="notice" id="qwen-rerun-banner">
  <h2>qwen rerun additions <span class="badge hot" style="background:#7c3aed;color:#fff">{len(new_slots)} new slots</span></h2>
  <p>Recovery rerun for MISS slots that failed the original gpt-5.5/gpt-image-2 stack
  due to gateway/Cloudflare 524 long-tail (verified 2026-05-15). Planner LLM switched
  to qwen3.6-max-preview on DashScope; image-gen kinds (portrait/anchor/edit)
  switched to wan2.7-image-pro on DashScope. Everything else identical to the
  2026-05-14 batch_18 run.</p>
  <ul>
    {''.join(f'<li><a href="#{s}">{s}</a></li>' for s in new_slots)}
  </ul>
</section>
"""

    # Append before </main>
    insert_marker = "</main>"
    insert_at = html.rfind(insert_marker)
    if insert_at < 0:
        # Fallback: just append to body
        html = html + banner + sections
    else:
        html = html[:insert_at] + banner + sections + html[insert_at:]

    index_html.write_text(html, encoding="utf-8")
    print(f"  appended {len(new_slots)} sections to {index_html.name}", flush=True)


def update_experiment_json(new_slots: list[str], dry: bool) -> None:
    if dry:
        print(f"  [dry] would update experiment.json entries", flush=True)
        return
    ep = BUNDLE / "experiment.json"
    data = json.loads(ep.read_text())
    existing = set(data.get("entry_ids", []) or [])
    for slot in new_slots:
        existing.add(slot)
    data["entry_ids"] = sorted(existing)
    data["entries"] = len(existing)
    data["last_qwen_rerun"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "slots_added": list(new_slots),
        "stack": "qwen3.6-max-preview-planner + wan2.7-image-pro",
        "reason": "gateway/CF 524 long-tail truncated gpt-5.5 planner + gpt-image-2",
    }
    ep.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  experiment.json updated: {len(existing)} entries total", flush=True)


def update_manifest_json(new_slots: list[str], slot_meta: dict[str, dict],
                         dry: bool) -> None:
    if dry:
        print(f"  [dry] would update manifest.json", flush=True)
        return
    mp = BUNDLE / "manifest.json"
    data = json.loads(mp.read_text())
    data.setdefault("qwen_rerun_slots", {})
    for slot in new_slots:
        data["qwen_rerun_slots"][slot] = slot_meta[slot]
    mp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  manifest.json updated: +{len(new_slots)} qwen_rerun_slots", flush=True)


def upload_to_oss(dry: bool) -> None:
    if dry:
        print("  [dry] would upload to OSS", flush=True)
        return
    mod_path = REPO / "experimental_baseline/tools/oss_preview.py"
    spec = importlib.util.spec_from_file_location("oss_preview", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("oss_preview spec failed")
    oss_preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oss_preview)  # type: ignore[attr-defined]

    bucket_name = "intern-data-wlcb"
    endpoint = "https://oss-cn-wulanchabu.aliyuncs.com"
    prefix = f"new-videos/{BUNDLE.name}/{time.strftime('%Y%m%d-%H%M%S')}-qwen-rerun"
    expires_s = 30 * 24 * 3600

    oss_preview.auto_load_project_env()
    bucket = oss_preview.make_bucket(endpoint, bucket_name)
    t0 = time.time()
    uploaded = oss_preview.upload_bundle(bucket, BUNDLE, prefix)
    print(f"  uploaded {len(uploaded)} files in {time.time()-t0:.1f}s", flush=True)

    signed = oss_preview.build_signed_urls(bucket, uploaded, expires_s=expires_s)

    # Rewrite index.html → index.oss.html with signed URLs
    html = (BUNDLE / "index.html").read_text(encoding="utf-8")
    for rel in sorted(signed, key=len, reverse=True):
        if rel in {"index.html", "index.oss.html"}:
            continue
        html = html.replace(f'src="{rel}"', f'src="{signed[rel]}"')
        html = html.replace(f'href="{rel}"', f'href="{signed[rel]}"')
        # Also handle <source src="...">
        html = html.replace(f'<source src="{rel}"', f'<source src="{signed[rel]}"')
    (BUNDLE / "index.oss.html").write_text(html, encoding="utf-8")
    # Re-upload (uploaded is rel→key dict); fresh dict picks up the new index.oss.html.
    uploaded2 = oss_preview.upload_bundle(bucket, BUNDLE, prefix)
    signed2 = oss_preview.build_signed_urls(bucket, uploaded2, expires_s=expires_s)
    index_oss_url = signed2.get("index.oss.html", "")

    (BUNDLE / "oss_upload_result.json").write_text(
        json.dumps({
            "bucket": bucket_name,
            "endpoint": endpoint,
            "prefix": prefix,
            "expires_s": expires_s,
            "uploaded_count": len(uploaded),
            "index_oss_url": index_oss_url,
            "cdn_ready_relative_entry": f"{prefix}/index.html",
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stack": "qwen3.6-max-preview-planner + wan2.7-image-pro",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  index.oss.html: {index_oss_url}", flush=True)


def main() -> int:
    args = parse_args()
    explicit = [s.strip() for s in args.slots.split(",") if s.strip()]

    print(f"[merge] target bundle: {BUNDLE.name}", flush=True)
    new = find_completed_new_slots(explicit)
    if not new:
        print("[merge] no new completed slots to add", flush=True)
        if args.upload and not args.dry_run:
            print("[merge] --upload requested without new slots; uploading current bundle anyway", flush=True)
            upload_to_oss(dry=False)
        return 0
    print(f"[merge] new slots to add: {len(new)}: {', '.join(new)}", flush=True)

    slot_meta: dict[str, dict] = {}
    for slot in new:
        print(f"[merge] copy {slot}", flush=True)
        slot_meta[slot] = copy_slot_assets(slot, dry=args.dry_run)

    merge_into_index(new, slot_meta, dry=args.dry_run)
    update_experiment_json(new, dry=args.dry_run)
    update_manifest_json(new, slot_meta, dry=args.dry_run)

    if args.upload and not args.dry_run:
        upload_to_oss(dry=False)
    elif not args.upload:
        print("[merge] local bundle updated. Pass --upload to push to OSS.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
