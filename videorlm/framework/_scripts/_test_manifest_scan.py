"""Smoke test: feed the manifest scanner a real completed run and verify
every demo-page-style asset comes back with browser-fetchable URLs.

Uses one of the already-rendered v2.4 cells as the job dir (no API keys
needed). Prints the manifest sections and a summary count so you can eye-
ball whether the page would render shots / portraits / anchors / segments
the way the showcase grid does.

Run::

    cd /mnt/workspace/akide/code/unirlm-02
    python3 -m videorlm.framework._scripts._test_manifest_scan
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
# Use a known-good completed cell as the source data
SOURCE = REPO / "videorlm" / "outputs" / "version_2.4" / "ex49_happyhorse"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    # symlink to keep it cheap (we never write back through here)
    dst.symlink_to(src)


def _populate_fake_job(job_dir: Path, source: Path) -> None:
    """Mirror the layout the manifest scanner expects."""
    # Top-level planning JSONs
    for name in ("skeleton.json", "planner.json", "render_plan.json"):
        s = source / name
        if s.exists():
            _link_or_copy(s, job_dir / name)
    # run/ tree
    run_src = source / "run"
    if not run_src.exists():
        print(f"[fatal] source has no run/: {source}")
        sys.exit(2)
    for sub in ("portraits", "locations", "props", "anchors", "segments", "bridges"):
        src = run_src / sub
        if not src.exists(): continue
        dst = job_dir / "run" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix in (".png", ".mp4"):
                _link_or_copy(f, dst / f.name)
    # final.mp4
    final_src = run_src / "final.mp4"
    if final_src.exists():
        _link_or_copy(final_src, job_dir / "run" / "final.mp4")


def main() -> int:
    if not SOURCE.exists():
        print(f"[fatal] source cell missing: {SOURCE}")
        return 2

    from videorlm.framework._scripts._web_server import _scan_manifest, JOBS_ROOT

    # Build a temp job dir under the same JOBS_ROOT so server-rooted paths
    # are stable (not strictly required since we call the scanner directly).
    jid = "test_manifest"
    job_dir = JOBS_ROOT / jid
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)

    print(f"populating fake job at {job_dir} from {SOURCE.name}")
    _populate_fake_job(job_dir, SOURCE)

    manifest = _scan_manifest(jid, job_dir)

    # Pretty-print sections
    print()
    print(f"=== manifest for {SOURCE.name} ===")
    print(f"  skeleton    : {manifest['skeleton']}")
    print(f"  planner     : {manifest['planner']}")
    print(f"  render_plan : {manifest['render_plan']}")
    print(f"  final       : {manifest['final']}")
    print(f"  shots       : {len(manifest['shots'])} items")
    for s in manifest["shots"][:3]:
        print(f"      - {s['id']}  ({s['duration_s']}s)  '{s['story_goal'][:60]}…'")
    if len(manifest["shots"]) > 3:
        print(f"      … + {len(manifest['shots']) - 3} more")

    counts = {}
    for kind in ("portraits", "locations", "props", "anchors", "segments", "bridges"):
        items = manifest.get(kind) or []
        counts[kind] = len(items)
        sample = items[0]["name"] if items else "(empty)"
        print(f"  {kind:11s}: {len(items):3d} items  (sample: {sample})")

    # Pass / fail summary
    failures = []
    if not manifest["shots"]: failures.append("no shots parsed (skeleton/planner.shots empty)")
    if counts["anchors"] == 0: failures.append("no anchors found")
    if counts["segments"] == 0: failures.append("no segments found")
    if not manifest["final"]: failures.append("final.mp4 URL missing")
    if not manifest["skeleton"]: failures.append("skeleton URL missing")
    if not manifest["planner"]: failures.append("planner URL missing")

    print()
    if failures:
        print(f"✗ {len(failures)} failure(s):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print("✓ manifest scan picked up everything the showcase grid needs")
    print(f"  page would render: {len(manifest['shots'])} shot cards,",
          f"{counts['portraits']} portraits,", f"{counts['anchors']} anchors,",
          f"{counts['segments']} segments + final.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
