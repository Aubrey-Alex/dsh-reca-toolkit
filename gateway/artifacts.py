"""Artifact manifest handling and safe publication helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_RELATIVE = "run/artifact_manifest.json"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MANIFEST_RELATIVE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        value = {}
    return value if isinstance(value, dict) else {}


def fallback_manifest(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Migration fallback for old runs created before ReCA manifest output."""
    run = run_dir / "run"
    candidates = (
        ("final_video", "run/final.mp4", "video/mp4"),
        ("plan", "planner.json", "application/json"),
        ("render_plan", "render_plan.json", "application/json"),
        ("summary", "run/summary.json", "application/json"),
        ("run_report", "run/run_report.json", "application/json"),
        ("events", "events.jsonl", "application/x-ndjson"),
        ("inputs", "input_manifest.json", "application/json"),
        ("contact_sheet", "run/contact.jpg", "image/jpeg"),
    )
    artifacts = []
    for kind, relative, mime in candidates:
        path = run_dir / relative
        artifacts.append({
            "kind": kind,
            "path": relative,
            "mime": mime,
            "status": "ready" if path.is_file() else "missing",
        })
    audit_path = run / "audit.json"
    artifacts.append({
        "kind": "audit",
        "path": "run/audit.json",
        "mime": "application/json",
        "status": "ready" if audit_path.is_file() else "skipped",
    })
    return {"run_id": run_id, "version": 1, "artifacts": artifacts}


def manifest_for_run(run_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    return manifest if manifest.get("artifacts") else fallback_manifest(run_dir, run_id)


def public_manifest(run_dir: Path, run_id: str, base_url: str) -> dict[str, Any]:
    manifest = manifest_for_run(run_dir, run_id)
    base_url = base_url.rstrip("/")
    result = dict(manifest)
    result["run_id"] = run_id
    published = []
    for item in manifest.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        relative = str(entry.get("path") or "").lstrip("/")
        entry["url"] = f"{base_url}/v1/runs/{run_id}/artifacts/{relative}"
        published.append(entry)
    result["artifacts"] = published
    return result
