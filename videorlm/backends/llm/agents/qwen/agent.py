"""DashScopeQwenAgent: OpenAI-compatible agent specialized for DashScope's
chat endpoint.

After the provider-layer refactor (2026-05-16), the multi-key rotation
machinery moved into ``OpenAICompatibleAgent._chat_create`` and is gated
by the ``PROVIDER_NAME`` class attribute. This subclass only needs to
set ``PROVIDER_NAME = "dashscope"`` to inherit dashscope KeyPool +
cooldown + EWMA + classifier — no per-class ``_chat_create`` override
required.

The legacy public symbol ``QwenAgent`` lives in this package's
``__init__`` as a factory that dispatches between this class (DashScope
base_url) and the plain ``OpenAICompatibleAgent`` (openai.com,
self-hosted gateways). Error messages and slot keys reflect the actual
dispatched class.
"""

from __future__ import annotations

from ..base import AgentCapabilities
from ..openai_compat.agent import OpenAICompatibleAgent
from .config import QwenConfig  # noqa: F401 — re-exported for legacy import paths


class DashScopeQwenAgent(OpenAICompatibleAgent):
    """OpenAI-compatible agent pinned at the DashScope key pool.

    Everything (streaming, multimodal, fork, AGENT_NAME slot key,
    multi-key rotation, cooldown, EWMA) inherits from
    ``OpenAICompatibleAgent`` — this subclass only flips
    ``PROVIDER_NAME`` to ``"dashscope"`` so the base's ``_chat_create``
    funnels through ``with_key("dashscope", ...)``.
    """

    AGENT_NAME = "qwen"
    FAMILY = "qwen-text"
    PROVIDER_NAME = "dashscope"

    def capabilities(self) -> AgentCapabilities:
        # Same slot key / cap as the base; override only the ``notes`` blurb.
        base_caps = super().capabilities()
        return AgentCapabilities(
            agent_name=base_caps.agent_name,
            family=self.FAMILY,
            max_concurrency=base_caps.max_concurrency,
            requests_per_minute=base_caps.requests_per_minute,
            notes=(
                f"DashScope OpenAI-compatible endpoint with multi-key rotation. "
                f"Server is stateless; full message list resent each turn. "
                f"slot_key={base_caps.agent_name!r} cap={base_caps.max_concurrency}"
            ),
        )


__all__ = ["DashScopeQwenAgent", "QwenConfig"]
