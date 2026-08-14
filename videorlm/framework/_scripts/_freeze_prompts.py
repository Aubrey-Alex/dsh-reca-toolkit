"""Snapshot prompt module sources at run start.

Why: ``planner.json`` is the durable artifact, but the prompts that
produced it live in Python modules that can drift between runs. Without a
snapshot, comparing two ``planner.json`` files across days is misleading
because the prompt strings the parent / SP / validator saw aren't
recoverable. ``freeze_prompts(out_dir)`` writes the 4 prompt module
sources into ``<out_dir>/_frozen_prompts/`` plus a ``freeze_meta.json``
with the timestamp + git SHA when available, so any artifact can be
traced back to the exact prompt set that produced it.

Call from ``run_planner()`` / ``_smoke.py`` at the start of a run.
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def _try_git_sha() -> str | None:
    """Return current HEAD SHA, or None if not inside a git repo (this
    research repo is not a git repo at the time of writing; the function
    still tries so the snapshot mechanism survives a future move into git)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        return out or None
    except Exception:
        return None


def freeze_prompts(out_dir: Path | str) -> None:
    """Write a snapshot of the 4 prompt module sources into
    ``<out_dir>/_frozen_prompts/`` plus a tiny meta file.

    Files written:
      - ``shot_planner.py``     — PARENT_SYSTEM_PROMPT module source
      - ``segment_planner.py``  — SP system + replan + user prompt formatters
      - ``validator_anchor.py`` — anchor validator prompts
      - ``validator_segment.py``— segment validator prompts
      - ``freeze_meta.json``    — {frozen_at, git_sha}
    """
    from videorlm.framework.shot_planner import prompts as p_shot
    from videorlm.framework.segment_planner import prompts as p_seg
    from videorlm.framework.validator.anchor import prompts as p_val_a
    from videorlm.framework.validator.segment import prompts as p_val_s

    frozen = Path(out_dir) / "_frozen_prompts"
    frozen.mkdir(parents=True, exist_ok=True)
    for mod, name in [
        (p_shot,  "shot_planner.py"),
        (p_seg,   "segment_planner.py"),
        (p_val_a, "validator_anchor.py"),
        (p_val_s, "validator_segment.py"),
    ]:
        (frozen / name).write_text(inspect.getsource(mod), encoding="utf-8")

    meta = {
        "frozen_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_sha": _try_git_sha(),
        "env_RECA_FORK_INHERIT_PAIRS": os.environ.get("RECA_FORK_INHERIT_PAIRS", ""),
    }
    (frozen / "freeze_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


__all__ = ["freeze_prompts"]
