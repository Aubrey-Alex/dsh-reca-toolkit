"""ReCA-owned run state and artifact manifest writer.

Gateway reads these files and projects them outward; it does not derive
business stages or audit state from log strings.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def state_path(out_dir: Path) -> Path:
    return out_dir / "run" / "reca_state.json"


def write_state(
    out_dir: Path,
    *,
    stage: str,
    state: str = "running",
    audit_state: str = "audit_pending",
    video_state: str = "pending",
    progress: float | None = None,
    **extra: Any,
) -> None:
    path = state_path(out_dir)
    current: dict[str, Any] = {}
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    current.update({
        "version": 1,
        "updated_at": time.time(),
        "state": state,
        "stage": stage,
        "audit_state": audit_state,
        "video_state": video_state,
    })
    if progress is not None:
        current["progress"] = round(max(0.0, min(1.0, progress)), 3)
    current.update(extra)
    _write(path, current)


def write_artifact_manifest(out_dir: Path, *, run_id: str = "") -> Path:
    run = out_dir / "run"
    audit = run / "audit.json"
    entries = [
        ("final_video", "run/final.mp4", "video/mp4"),
        ("plan", "planner.json", "application/json"),
        ("render_plan", "render_plan.json", "application/json"),
        ("audit", "run/audit.json", "application/json"),
        ("events", "events.jsonl", "application/x-ndjson"),
        ("inputs", "input_manifest.json", "application/json"),
        ("summary", "run/summary.json", "application/json"),
        ("run_report", "run/run_report.json", "application/json"),
        ("contact_sheet", "run/contact.jpg", "image/jpeg"),
    ]
    artifacts = []
    audit_status = "missing"
    if audit.is_file():
        try:
            audit_state = json.loads(audit.read_text(encoding="utf-8")).get("state")
        except (OSError, ValueError):
            audit_state = None
        audit_status = "skipped" if audit_state == "audit_skipped" else "ready"
    for kind, relative, mime in entries:
        path = out_dir / relative
        if kind == "audit":
            status = audit_status
        else:
            status = "ready" if path.is_file() else "missing"
        artifacts.append({"kind": kind, "path": relative, "mime": mime, "status": status})
    manifest = {"version": 1, "run_id": run_id, "generated_at": time.time(), "artifacts": artifacts}
    path = run / "artifact_manifest.json"
    _write(path, manifest)
    return path


def write_audit_report(out_dir: Path, *, state: str, details: dict[str, Any] | None = None) -> Path:
    path = out_dir / "run" / "audit.json"
    _write(path, {
        "version": 1,
        "state": state,
        "updated_at": time.time(),
        "details": details or {},
    })
    return path


def write_run_report(out_dir: Path, *, state: str, details: dict[str, Any] | None = None) -> Path:
    path = out_dir / "run" / "run_report.json"
    _write(path, {
        "version": 1,
        "state": state,
        "updated_at": time.time(),
        "details": details or {},
    })
    return path


def write_contact_sheet(out_dir: Path) -> Path | None:
    """Create a small inspection sheet when ffmpeg is available."""
    video = out_dir / "run" / "final.mp4"
    output = out_dir / "run" / "contact.jpg"
    if not video.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                "-vf", "fps=1/3,scale=320:-1,tile=4x2:padding=4:margin=4",
                "-frames:v", "1", str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return output if result.returncode == 0 and output.is_file() else None
