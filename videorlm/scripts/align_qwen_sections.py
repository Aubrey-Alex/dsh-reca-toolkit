"""Align the qwen-rerun sections with the original v2.4 batch_18 sections.

What this fixes (response to user feedback after the first append-merge):
  1. The TOC grid at the top of index.html had no cards for the 18 new
     rerun slots — make them clickable.
  2. The appended sections hardcoded ``1280x720`` for wan resolution, but
     actual final.mp4 is 1920x1080 (wan2.7-r2v PROVIDER_SPEC declares
     720p but DashScope actually renders 1080p — provider-side lift not
     yet reflected in the framework's `supports_resolutions` tuple).
  3. The appended sections were minimal (video + 3 meta chips + tiny
     "Recovery context" blurb). The originals have:
       - first-shot planner summary (pre)
       - anchors grid
       - portrait/location/prop reference grids
       - segment tail-frame grid
       - per-cell render prompts + refs
     Mirror that full structure so the rerun slots look the same.

What this writes:
  - Copies anchors/, portraits/, locations/, props/, segments/*.tail.png
    + planner.json + render_plan.json into ``assets/<slot>/`` (these are
    referenced by the full-format <details> sections).
  - Replaces the 18 minimal sections in index.html with full-format
    ones. Idempotent: looks for existing <section id="<slot>"> and
    swaps it out.
  - Inserts 18 TOC cards into the .toc grid right after the last
    existing card, sorted by slot id.
  - Probes each final.mp4 via ffprobe for the actual (width, height,
    duration) so the meta line is honest.

After this runs, follow up with append_to_v24_bundle.py --upload (it
will detect no new copies needed but re-upload + re-sign OSS URLs and
regenerate index.oss.html).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path("/mnt/workspace/akide/code/unirlm-02")
RUN_ROOT = REPO / "videorlm/outputs/version_2.4"
BUNDLE = REPO / "outputs/new_videos/20260514-version-2-4-batch-18-wan-vs-happyhorse"

# Slots added by the qwen rerun (must match append_to_v24_bundle.py).
RERUN_SLOTS = (
    "ex01_wan", "ex01_happyhorse",
    "ex07_wan", "ex13_happyhorse", "ex17_happyhorse",
    "ex19_wan", "ex23_wan", "ex29_wan",
    "ex31_happyhorse", "ex32_happyhorse",
    "ex37_wan", "ex38_wan", "ex39_wan", "ex40_wan",
    "ex41_wan", "ex42_wan", "ex43_wan", "ex44_wan",
)

# Backend label for the "wan_r2v_1080p" / "happyhorse_r2v_1080p" chip.
def backend_chip(slot: str, width: int) -> str:
    backend = "wan" if slot.endswith("_wan") else "happyhorse"
    res_tag = "1080p" if width >= 1900 else "720p"
    return f"{backend}_r2v_{res_tag}"


def ffprobe(mp4: Path) -> tuple[int, int, float]:
    """Return (width, height, duration_s) — fall back to 0s on failure."""
    if not mp4.exists():
        return (0, 0, 0.0)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-show_entries", "stream=width,height", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1",
             str(mp4)],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        w = h = 0
        d = 0.0
        for line in out.splitlines():
            if line.startswith("width="):
                w = int(line.partition("=")[2] or 0)
            elif line.startswith("height="):
                h = int(line.partition("=")[2] or 0)
            elif line.startswith("duration="):
                d = float(line.partition("=")[2] or 0)
        return (w, h, d)
    except Exception as exc:
        print(f"  ffprobe fail on {mp4}: {exc}", flush=True)
        return (0, 0, 0.0)


def copy_full_assets(slot: str) -> None:
    """Copy anchors/portraits/locations/props/segments(.tail.png only) +
    planner.json + render_plan.json into the bundle."""
    src = RUN_ROOT / slot
    dst = BUNDLE / "assets" / slot
    dst.mkdir(parents=True, exist_ok=True)
    # JSON plans
    for name in ("planner.json", "render_plan.json", "skeleton.json"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    # PNG grids — anchors / portraits / locations / props
    for subdir in ("anchors", "portraits", "locations", "props"):
        src_dir = src / "run" / subdir
        if not src_dir.exists():
            continue
        dst_dir = dst / subdir
        dst_dir.mkdir(exist_ok=True)
        for png in src_dir.glob("*.png"):
            target = dst_dir / png.name
            if not target.exists() or target.stat().st_mtime < png.stat().st_mtime:
                shutil.copy2(png, target)
    # Segment tail frames only (mp4s themselves are HUGE; OSS handles the
    # concat'd final.mp4 separately).
    seg_src = src / "run" / "segments"
    if seg_src.exists():
        seg_dst = dst / "segments"
        seg_dst.mkdir(exist_ok=True)
        for tail in seg_src.glob("*.tail.png"):
            target = seg_dst / tail.name
            if not target.exists() or target.stat().st_mtime < tail.stat().st_mtime:
                shutil.copy2(tail, target)


def render_section_html(slot: str, dur_s: float, width: int, height: int) -> str:
    """Build the full-format <section class="run"> mirroring the originals."""
    backend = "wan" if slot.endswith("_wan") else "happyhorse"
    res_tag = "1080P" if width >= 1900 else ("720P" if width >= 1200 else f"{width}P")
    ex_num = slot.split("_")[0].replace("ex", "")
    final_mp4 = BUNDLE / "assets" / slot / "final.mp4"
    size_mb = (final_mp4.stat().st_size / 1024 / 1024) if final_mp4.exists() else 0

    # The two ex01 slots were launched before run_ex01_qwen.sh got upgraded
    # to wan2.7-image-pro — they actually used wan2.7-image (non-pro) for
    # portrait/anchor and qwen-image-2.0-pro for image_edit (and image_edit
    # was broken with "url error" so all validator-repair calls failed —
    # the anchors that landed in this slot are the unrepaired T2I outputs).
    if slot in {"ex01_wan", "ex01_happyhorse"}:
        image_label = "wan2.7-image (T2I) · qwen-image-2.0-pro (edit, framework-bug, all repairs no-op)"
    else:
        image_label = "wan2.7-image-pro"

    # Load plan data
    rp_path = BUNDLE / "assets" / slot / "render_plan.json"
    sk_path = BUNDLE / "assets" / slot / "skeleton.json"
    render_plan = {}
    skeleton = {}
    if rp_path.exists():
        try:
            render_plan = json.loads(rp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  {slot}: render_plan.json parse fail: {exc}", flush=True)
    if sk_path.exists():
        try:
            skeleton = json.loads(sk_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  {slot}: skeleton.json parse fail: {exc}", flush=True)

    shots = render_plan.get("shots") or skeleton.get("shots") or []
    anchors = render_plan.get("boundary_anchors") or skeleton.get("boundarys", {}).get("boundary_anchors", []) or []
    # ``segments`` is a dict keyed by seg_id in render_plan.json; normalise to a list.
    segs_raw = render_plan.get("segments") or {}
    if isinstance(segs_raw, dict):
        segments_in_plan = list(segs_raw.values())
    elif isinstance(segs_raw, list):
        segments_in_plan = segs_raw
    else:
        segments_in_plan = []
    n_shots = len(shots)
    n_anchors = len(anchors)
    n_segs = len(segments_in_plan)
    durations = [int(s.get("duration_s", 0)) for s in shots]

    # File system lists for the asset grids
    def list_png(subdir: str) -> list[str]:
        d = BUNDLE / "assets" / slot / subdir
        if not d.exists():
            return []
        return sorted(p.name for p in d.glob("*.png") if not p.name.endswith(".tail.png"))

    anchor_files = list_png("anchors")
    portrait_files = list_png("portraits")
    location_files = list_png("locations")
    prop_files = list_png("props")
    seg_dir = BUNDLE / "assets" / slot / "segments"
    seg_tails = sorted(p.name for p in seg_dir.glob("*.tail.png")) if seg_dir.exists() else []

    def grid(subdir: str, files: list[str]) -> str:
        items = "".join(
            f'<figure><img data-src="assets/{slot}/{subdir}/{f}" loading="lazy"/>'
            f'<figcaption>{Path(f).stem}</figcaption></figure>'
            for f in files
        )
        return f'<div class="asset-grid">{items}</div>'

    # First-shot summary collapsible
    first_shot_json = "{}"
    if shots:
        first_shot_json = json.dumps(shots[0], ensure_ascii=False, indent=2)

    # Render prompts list — portraits + locations + zones + props + anchors + segments
    prompt_blocks: list[str] = []

    def add_prompt(title: str, subtype: str, prompt: str,
                   neg: str = "", refs: Any = None) -> None:
        if not prompt:
            return
        refs_html = ""
        if refs:
            try:
                refs_str = json.dumps(refs, ensure_ascii=False).replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                refs_html = f'<p class="refs">Refs: {refs_str}</p>'
            except Exception:
                pass
        neg_html = f'<p class="neg">Negative: {neg}</p>' if neg else ""
        prompt_blocks.append(
            f'<div class="prompt"><b>{title} <span>{subtype}</span></b>'
            f'<p>{prompt}</p>{neg_html}{refs_html}</div>'
        )

    # Portraits
    for pid, p in (render_plan.get("portrait_plan") or {}).items():
        add_prompt(pid, "portrait", p.get("prompt", ""), p.get("negative_prompt", ""))
    # Locations + zones
    for lid, loc in (render_plan.get("location_plan") or {}).items():
        add_prompt(lid, "location", loc.get("prompt", ""), loc.get("negative_prompt", ""))
        for zone in (loc.get("key_zones") or []):
            add_prompt(zone.get("id", "zone"), "zone", zone.get("prompt", ""),
                       zone.get("negative_prompt", ""))
    # Props
    for ploc_id, loc in (render_plan.get("location_plan") or {}).items():
        for prop_id, prop in (loc.get("props") or {}).items():
            add_prompt(prop_id, "prop", prop.get("prompt", ""), prop.get("negative_prompt", ""))
    for prop_id, prop in (render_plan.get("prop_plan") or {}).items():
        add_prompt(prop_id, "prop", prop.get("prompt", ""), prop.get("negative_prompt", ""))
    # Boundary anchors
    for anc in anchors:
        add_prompt(anc.get("id", "anchor"), "anchor", anc.get("prompt", ""),
                   anc.get("negative_prompt", ""),
                   refs=anc.get("reference_inputs") or anc.get("refs"))
    # Segments — render_plan stores each as {id, shot_id, segment_request: {prompt, negative_prompt, ...}, reference_inputs}.
    for seg in segments_in_plan:
        if not isinstance(seg, dict):
            continue
        shot_id = seg.get("shot_id") or seg.get("shot") or "?"
        sr = seg.get("segment_request") or {}
        prompt = sr.get("prompt") if isinstance(sr, dict) else None
        neg = sr.get("negative_prompt", "") if isinstance(sr, dict) else ""
        if not prompt:
            prompt = seg.get("prompt", "")
        add_prompt(seg.get("id", "segment"), f"segment · shot={shot_id}",
                   prompt or "", neg or "",
                   refs=seg.get("reference_inputs") or seg.get("refs"))

    prompts_html = "".join(prompt_blocks)

    # Compose final section
    return (
        f'<section class="run" id="{slot}">'
        f'<div class="run-head">'
        f'<div>'
        f'<h2>{slot} <span class="badge hot">{res_tag}</span> '
        f'<span class="badge hot" style="background:#7c3aed;color:#fff">qwen rerun</span></h2>'
        f'<p>final.mp4 · {width}x{height} · {dur_s:.1f}s · {size_mb:.1f} MB · '
        f'{n_shots} shots · {n_anchors} anchors · {n_segs} segments</p>'
        f'<p class="path">videorlm/outputs/version_2.4/{slot} · planner=qwen3.6-max-preview · image={image_label}</p>'
        f'</div>'
        f'<a class="top" href="#top">Back to top</a>'
        f'</div>'
        f'<video class="main-video" controls preload="none" loading="lazy" poster="assets/{slot}/contact.jpg">'
        f'<source src="assets/{slot}/final.mp4" type="video/mp4"></video>'
        f'<div class="meta">'
        f'<span>story: example_{ex_num}.txt</span>'
        f'<span>backend: {backend} · {width}x{height}</span>'
        f'<span>plan: {n_shots} shots, {n_segs} segs, duration {durations}s</span>'
        f'</div>'
        f'<details><summary>First shot summary (from planner)</summary><pre>{first_shot_json}</pre></details>'
        f'<details><summary>Anchors ({len(anchor_files)})</summary>{grid("anchors", anchor_files)}</details>'
        f'<details><summary>Portrait references ({len(portrait_files)})</summary>{grid("portraits", portrait_files)}</details>'
        f'<details><summary>Location references ({len(location_files)})</summary>{grid("locations", location_files)}</details>'
        f'<details><summary>Prop references ({len(prop_files)})</summary>{grid("props", prop_files)}</details>'
        f'<details><summary>Segment tail frames ({len(seg_tails)})</summary>{grid("segments", seg_tails)}</details>'
        f'<details><summary>Render prompts / refs ({len(prompt_blocks)})</summary><div class="prompts">{prompts_html}</div></details>'
        f'</section>'
    )


def build_toc_card(slot: str, dur_s: float, width: int, height: int) -> str:
    chip = backend_chip(slot, width)
    return (
        f'<a class="toc-card" href="#{slot}"><b>{slot}</b>'
        f'<span>{chip} · {width}x{height} · {dur_s:.1f}s · '
        f'<span style="color:#a78bfa">qwen rerun</span></span></a>'
    )


def replace_or_append_section(html: str, slot: str, new_section: str) -> str:
    """Replace existing <section id="<slot>"> block or append before </main>."""
    pattern = re.compile(
        rf'<section class="run" id="{re.escape(slot)}"[\s\S]*?</section>',
        re.DOTALL,
    )
    if pattern.search(html):
        html = pattern.sub(new_section, html, count=1)
    else:
        idx = html.rfind("</main>")
        if idx >= 0:
            html = html[:idx] + new_section + html[idx:]
        else:
            html = html + new_section
    return html


def insert_toc_cards(html: str, cards: list[str]) -> str:
    """Insert TOC cards into the .toc grid. Idempotent: drops any existing
    qwen-rerun cards (by slot id) before inserting fresh ones."""
    # Drop any pre-existing rerun cards
    for slot in RERUN_SLOTS:
        pattern = re.compile(rf'<a class="toc-card" href="#{re.escape(slot)}">[\s\S]*?</a>')
        html = pattern.sub("", html)
    # The TOC grid is a <nav class="toc"> ... </nav> (or <div class="toc"> on
    # older bundles). Insert just before the closing tag.
    for open_tag, close_tag in (
        ('<nav class="toc">', "</nav>"),
        ('<div class="toc">', "</div>"),
    ):
        idx = html.find(open_tag)
        if idx < 0:
            continue
        close_idx = html.find(close_tag, idx + len(open_tag))
        if close_idx < 0:
            continue
        return html[:close_idx] + "".join(cards) + html[close_idx:]
    print("  WARN: no .toc grid found", flush=True)
    return html


def main() -> int:
    print(f"[align] target bundle: {BUNDLE.name}", flush=True)

    metrics: dict[str, tuple[int, int, float]] = {}
    for slot in RERUN_SLOTS:
        final_mp4 = BUNDLE / "assets" / slot / "final.mp4"
        if not final_mp4.exists():
            print(f"  skip {slot}: no final.mp4 in bundle", flush=True)
            continue
        copy_full_assets(slot)
        w, h, d = ffprobe(final_mp4)
        metrics[slot] = (w, h, d)
        print(f"  {slot}: {w}x{h} · {d:.1f}s · assets copied", flush=True)

    if not metrics:
        print("[align] no slots to align", flush=True)
        return 1

    for index_name in ("index.html",):
        path = BUNDLE / index_name
        html = path.read_text(encoding="utf-8")
        # Re-render every rerun section
        for slot, (w, h, d) in metrics.items():
            new_section = render_section_html(slot, d, w, h)
            html = replace_or_append_section(html, slot, new_section)
        # Insert TOC cards (sorted by slot id)
        cards = [
            build_toc_card(slot, d, w, h)
            for slot, (w, h, d) in sorted(metrics.items())
        ]
        html = insert_toc_cards(html, cards)
        path.write_text(html, encoding="utf-8")
        print(f"  wrote {path.name} ({len(html)} chars)", flush=True)

    print("[align] DONE. Run append_to_v24_bundle.py --upload to re-publish to OSS.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
