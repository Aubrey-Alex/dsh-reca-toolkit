"""Recovery rules for Gateway restarts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTIVE_STATES = {"queued", "running", "cancelling"}


def recover_unfinished_runs(runs_root: Path) -> list[str]:
    """Mark unfinished persisted jobs interrupted without resubmitting work."""
    recovered: list[str] = []
    if not runs_root.is_dir():
        return recovered
    for state_path in runs_root.glob("*/state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(state, dict) or state.get("state") not in ACTIVE_STATES:
            continue
        state["state"] = "interrupted"
        # Keep the public Gateway projection in lockstep with the lifecycle
        # state. ReCA's own state remains untouched and is the business source
        # of truth for resume decisions.
        state["gateway_state"] = "interrupted"
        state["stage"] = "interrupted"
        state["error"] = "Gateway restarted before the run reached a terminal state"
        state["recovered_at"] = __import__("time").time()
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(state_path)
        recovered.append(str(state.get("run_id") or state_path.parent.name))
    return recovered
