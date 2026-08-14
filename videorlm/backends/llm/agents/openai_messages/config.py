"""Anthropic Messages API agent config — extends ``AgentConfig`` with
``thinking_budget_tokens`` (Anthropic-only).

Used by ``OpenAIMessagesAgent``. The ``thinking_budget_tokens`` field
was previously squatting on ``QwenConfig`` (cross-provider kitchen
sink); the 2026-05-16 refactor moved it here so each provider's config
only carries its own knobs.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import AgentConfig


@dataclass(frozen=True)
class OpenAIMessagesConfig(AgentConfig):
    """Anthropic Messages API agent config.

    Only delta vs ``AgentConfig``: ``thinking_budget_tokens``.

    When ``thinking_budget_tokens > 0`` the agent's
    ``/v1/messages`` POST includes::

        "thinking": {"type": "enabled", "budget_tokens": <n>}

    Default 0 disables Anthropic extended thinking.
    """

    thinking_budget_tokens: int = 0


__all__ = ["OpenAIMessagesConfig"]
