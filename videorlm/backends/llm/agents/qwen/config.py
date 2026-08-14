"""DashScope Qwen-specific agent config — extends ``AgentConfig`` with
``enable_thinking`` (qwen-3.x feature) and re-pins ``model`` to the
user-default ``qwen3.6-max-preview``.

Used by ``DashScopeQwenAgent`` (and any caller wanting qwen-flavored
defaults). For generic OpenAI-compat use, see
``backends.llm.agents.config.AgentConfig`` directly. For Anthropic
Messages API, see ``backends.llm.agents.openai_messages.config``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import AgentConfig


@dataclass(frozen=True)
class QwenConfig(AgentConfig):
    """DashScope Qwen agent config. Only delta vs ``AgentConfig``:

      - ``model`` defaults to ``"qwen3.6-max-preview"`` (user-pinned
        planner default as of 2026-04-28)
      - ``enable_thinking`` — when True, ``chat.completions`` passes
        ``extra_body={"enable_thinking": True}`` so qwen-3.x emits a
        hidden reasoning phase + reasoning_content delta stream before
        final content. We drop reasoning_content silently in the
        streaming loop (visible benefit: more deliberative planner
        output, matches gpt-5.5 reasoning-tier quality).
    """

    model: str = "qwen3.6-max-preview"
    enable_thinking: bool = False


__all__ = ["QwenConfig"]
