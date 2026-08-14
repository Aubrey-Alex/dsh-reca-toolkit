from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodexConfig:
    """Immutable configuration for a `CodexAgent`."""

    binary_path: str = "codex-acp"
    cwd: Path = field(default_factory=Path.cwd)
    env: Mapping[str, str] = field(default_factory=dict)
    request_timeout_s: float = 600.0
    startup_timeout_s: float = 30.0
    max_concurrency: int = 8
    # Optional system instructions seeded on the first session.new turn. Codex
    # also reads `AGENTS.md` / `CLAUDE.md` from `cwd` natively — prefer those
    # for static project context and use this knob only for dynamic, per-agent
    # instructions.
    system_prompt: str | None = None

    def with_overrides(self, **kwargs: Any) -> CodexConfig:
        return replace(self, **kwargs)
