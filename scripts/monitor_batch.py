#!/usr/bin/env python3
"""Monitor a ReCA Gateway batch and collect completed videos by title."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _gateway_status(base_url: str, run_id: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/v1/runs/{run_id}", timeout=15
    ) as response:
        value = json.loads(response.read().decode("utf-8"))
    return value if isinstance(value, dict) else {}


def _collect_final(batch_dir: Path, run: dict[str, Any]) -> str | None:
    source = Path(str(run.get("path") or "")) / "run" / "final.mp4"
    if not source.is_file():
        return None
    finals_dir = batch_dir / "finals"
    finals_dir.mkdir(parents=True, exist_ok=True)
    label = str(run.get("label") or run["run_id"])
    title = label.split("_", 1)[1] if "_" in label else label
    destination = finals_dir / f"{int(run.get('index', 0)):02d}_{title}.mp4"
    if destination.is_symlink() and destination.resolve() != source.resolve():
        destination.unlink()
    if not destination.exists():
        destination.symlink_to(source)
    return str(destination.absolute())


def monitor(manifest_path: Path, base_url: str, interval_s: float) -> int:
    batch_dir = manifest_path.parent
    status_path = batch_dir / "batch_status.json"
    while True:
        manifest = _read_json(manifest_path)
        terminal = 0
        succeeded = 0
        for run in manifest.get("runs", []):
            if not isinstance(run, dict) or not run.get("run_id"):
                continue
            try:
                status = _gateway_status(base_url, str(run["run_id"]))
                run.update({
                    "state": status.get("state"),
                    "stage": status.get("stage"),
                    "reca_stage": status.get("reca_stage"),
                    "audit_state": status.get("audit_state"),
                    "video_state": status.get("video_state"),
                    "final_video": status.get("final_video"),
                    "error": status.get("error"),
                })
            except (OSError, ValueError, urllib.error.URLError) as exc:
                run["monitor_error"] = f"{type(exc).__name__}: {exc}"
            state = str(run.get("state") or "")
            if state in TERMINAL_STATES:
                terminal += 1
            if state == "succeeded":
                succeeded += 1
                run["collected_final"] = _collect_final(batch_dir, run)

        total = len(manifest.get("runs", []))
        snapshot = {
            "batch_id": manifest.get("batch_id"),
            "updated_at": time.time(),
            "total": total,
            "terminal": terminal,
            "succeeded": succeeded,
            "runs": manifest.get("runs", []),
        }
        _write_json(manifest_path, manifest)
        _write_json(status_path, snapshot)
        print(
            f"batch={manifest.get('batch_id')} terminal={terminal}/{total} "
            f"succeeded={succeeded}/{total}",
            flush=True,
        )
        if total and terminal == total:
            return 0 if succeeded == total else 1
        time.sleep(max(5.0, interval_s))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:8787")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    return monitor(args.manifest.resolve(), args.gateway, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
