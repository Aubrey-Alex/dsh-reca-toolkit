#!/usr/bin/env python3
"""Materialize a static replay bundle from a real Gateway run."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("demo"))
    args = ap.parse_args()
    run = args.run_dir.resolve()
    out = args.output.resolve()
    manifest = run / "run" / "replay_manifest.json"
    if not manifest.exists():
        raise SystemExit(f"missing replay manifest: {manifest}")
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, out / "data" / "replay_manifest.json")
    assets = {
        "final.mp4": run / "run" / "final.mp4",
        "first-frame.png": run / "run" / "inputs" / "first_frame.png",
    }
    for name, source in assets.items():
        if source.exists():
            shutil.copy2(source, out / "assets" / name)
    print(json.dumps({"output": str(out), "run_id": run.name, "files": sorted(p.name for p in (out / "assets").iterdir())}))


if __name__ == "__main__":
    main()
