"""``RunContext`` — the per-run object threaded through every Stage / Callback.

What it carries:
  - run-level config (seed, out_dir, validate / validate_segments toggles)
  - the callback fan-out (logs / metrics / observability)
  - a tiny inter-stage scratchpad (dict) so a stage can stash something
    later stages may peek at without changing the formal output schema
  - the start ts so any sink can compute wall-clock dt against it

Why a separate file: ``stage.py``, ``parsers.py`` (via the on_parse_fail
bridge), the validator state machine, and pipeline_v2 all import it.
Keeping it free of heavy deps (no schemas, no agents) avoids the
circular import maze the legacy ``pipeline.py`` accumulated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .callbacks import Callback, MultiCallback, default_callbacks


@dataclass
class RunContext:
    """One ``RunContext`` per pipeline invocation. Shared by every stage.

    Stages MUST NOT mutate ``config`` / ``out_dir`` mid-run; that breaks
    the "stages are independently reorderable" invariant. The
    ``scratchpad`` is the only mutable place — use it for cached anchors,
    intermediate URLs, etc.
    """

    out_dir: Path
    seed: int = 0
    enable_anchor_validator: bool = True
    enable_segment_validator: bool = True
    max_repair_attempts: int = 2

    callback: Callback = field(default_factory=MultiCallback)
    started_at: float = field(default_factory=time.time)
    scratchpad: dict[str, Any] = field(default_factory=dict)

    # Tag surfaced into summary.json so downstream OSS / bundle tools
    # can distinguish runs from different image-gen / planner stacks.
    backend_tag: str = ""

    @classmethod
    def for_run(
        cls,
        out_dir: str | Path,
        *,
        seed: int = 0,
        enable_anchor_validator: bool = True,
        enable_segment_validator: bool = True,
        max_repair_attempts: int = 2,
        callback: Callback | None = None,
        backend_tag: str = "",
    ) -> "RunContext":
        """Construct a ready-to-use context. Default callbacks =
        stdout + JSONL trace + in-memory metrics."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return cls(
            out_dir=out,
            seed=seed,
            enable_anchor_validator=enable_anchor_validator,
            enable_segment_validator=enable_segment_validator,
            max_repair_attempts=max_repair_attempts,
            callback=callback or default_callbacks(out),
            backend_tag=backend_tag,
        )

    def elapsed_s(self) -> float:
        return time.time() - self.started_at


__all__ = ["RunContext"]
