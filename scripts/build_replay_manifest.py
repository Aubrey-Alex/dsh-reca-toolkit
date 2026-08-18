#!/usr/bin/env python3
"""Build a redacted, replay-friendly index for one completed Director run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            events.append(value)
    return events


def classify_log(text: str) -> str | None:
    patterns = (
        ("planning", ("plan_skeleton", "plan_segments_all", "planner")),
        ("assets", ("images-dag", "asset_generation", "portrait", "anchor")),
        ("generation", ("segments", "segment-trace", "wan3.0")),
        ("audit", ("validator", "validate", "audit", "repair")),
        ("concat", ("concat", "contact sheet")),
    )
    lowered = text.lower()
    for phase, markers in patterns:
        if any(marker.lower() in lowered for marker in markers):
            return phase
    return None


def shot_summary(shot: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": shot.get("id"),
        "intent": shot.get("intent") or shot.get("summary") or shot.get("description"),
        "start_state": shot.get("start_state"),
        "end_state": shot.get("end_state"),
    }


def segment_summary(segment_id: str, segment: dict[str, Any]) -> dict[str, Any]:
    request = segment.get("segment_request") or {}
    return {
        "id": segment_id,
        "shot_id": segment.get("shot_id"),
        "index": segment.get("segment_index_in_shot"),
        "duration_s": request.get("duration_s", segment.get("duration_s")),
        "prompt": request.get("prompt") or segment.get("prompt"),
    }


def build_manifest(run_dir: Path) -> dict[str, Any]:
    request = read_json(run_dir / "request.json", {})
    state = read_json(run_dir / "run" / "reca_state.json", {})
    planner = read_json(run_dir / "planner.json", {})
    render_plan = read_json(run_dir / "render_plan.json", {})
    audit = read_json(run_dir / "run" / "audit.json", {})
    events = read_events(run_dir / "events.jsonl")
    input_manifest = read_json(run_dir / "input_manifest.json", {})

    timeline: list[dict[str, Any]] = []
    last_phase: str | None = None
    for event in events:
        text = event.get("text", "")
        phase = classify_log(text)
        if phase and phase != last_phase:
            timeline.append({
                "ts": event.get("ts"),
                "phase": phase,
                "label": phase.replace("_", " "),
                "source": "reca_event",
                "detail": text,
            })
            last_phase = phase

    story = request.get("story") if isinstance(request, dict) else None
    options = request.get("options") if isinstance(request, dict) else {}
    return {
        "version": 1,
        "run_id": state.get("run_id") or run_dir.name,
        "recording": {
            "kind": "real_dsh_reca_run",
            "raw_event_count": len(events),
            "source": "Gateway events + ReCA planner/render/audit artifacts",
        },
        "dsh": {
            "user_story": story,
            "tool": "reca_create_video",
            "options": options,
            "result_state": state.get("state"),
        },
        "reca": {
            "stage": state.get("stage"),
            "audit_state": state.get("audit_state"),
            "video_state": state.get("video_state"),
            "shot_count": len(render_plan.get("shots") or planner.get("shots") or []),
            "segment_count": len(render_plan.get("segments") or {}),
            "asset_count": state.get("asset_count"),
            "audit": audit,
        },
        "inputs": input_manifest,
        "shots": [shot_summary(s) for s in (planner.get("shots") or []) if isinstance(s, dict)],
        "segments": [
            segment_summary(segment_id, segment)
            for segment_id, segment in (render_plan.get("segments") or {}).items()
            if isinstance(segment, dict)
        ],
        "timeline": timeline,
        "artifacts": {
            "first_frame": "input_manifest.json",
            "planner": "planner.json",
            "render_plan": "render_plan.json",
            "audit": "run/audit.json",
            "final_video": "run/final.mp4",
            "manifest": "run/artifact_manifest.json",
            "events": "events.jsonl",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.run_dir / "run" / "replay_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(args.run_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
