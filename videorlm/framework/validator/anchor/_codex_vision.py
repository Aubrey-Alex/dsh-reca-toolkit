"""Codex-acp wrapper for validator vision_call (Phase 2).

Replaces Phase 1 direct OpenAI SDK (`_gpt55_vision.py`) with
CodexAgent + image content blocks via ACP protocol. Backing model is
whatever `~/.codex/config.toml` configures (currently gpt-5.5 via
OpenAI-compatible gateway in this environment).

Each `vision_call(sys, user, img_url)` invocation spawns a fresh
codex-acp child process (5-10s startup overhead). For batch validation
this is fine; for high-throughput pipelines, swap to a session-pool
wrapper that reuses one CodexAgent across many calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def make_codex_vision_call(
    *,
    binary_path: str = "codex-acp",
    cwd: Path | None = None,
    request_timeout_s: float = 300.0,
) -> Callable[[str, str, str], str]:
    """Return a vision_call(system_prompt, user_prompt, image_url) -> str.

    Implementation:
      1. Construct a fresh CodexConfig with the system_prompt seeded.
      2. Open CodexAgent → spawns codex-acp child + opens ACP session.
      3. Call agent.prompt_with_images(user_prompt, [image_url])
         → fetches URL, base64-encodes, sends as ACP image content block.
      4. Returns assistant text reply, then closes the agent (subprocess
         reaped, session torn down).

    The returned callable matches validator.validate_anchor's
    vision_call contract; drop-in replacement for `make_gpt55_vision_call`.
    """
    if cwd is None:
        cwd = Path.cwd()

    def vision_call(system_prompt: str, user_prompt: str, image_url: str) -> str:
        # Lazy import: keeps validator package importable even without
        # codex-acp binary installed (Phase 1 callers don't need it).
        from videorlm.backends.llm.agents import CodexAgent, CodexConfig

        cfg = CodexConfig(
            binary_path=binary_path,
            cwd=cwd,
            system_prompt=system_prompt,
            request_timeout_s=request_timeout_s,
        )
        with CodexAgent(cfg) as agent:
            return agent.prompt_with_images(user_prompt, [image_url])

    return vision_call


__all__ = ["make_codex_vision_call"]
