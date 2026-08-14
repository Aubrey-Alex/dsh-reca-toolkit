"""Backwards-compatible facade for the legacy ``QwenAgent`` import.

``QwenAgent`` is a factory: returns a ``DashScopeQwenAgent`` for
DashScope / Aliyun base_urls, or a plain ``OpenAICompatibleAgent`` for
any other endpoint (gpt-5.5 (OpenAI-compat), openai.com, ...). Error messages
and slot keys reflect the actual dispatched class — so a gateway 503
logs as ``OpenAICompatibleAgent.prompt failed`` rather than the
previously-misleading ``QwenAgent.prompt failed``.

Existing callers (``with QwenAgent(cfg) as parent: ...``) keep working
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..openai_compat.agent import OpenAICompatibleAgent
from .agent import DashScopeQwenAgent
from .config import QwenConfig

if TYPE_CHECKING:
    from ..base import Agent, AgentState


def _is_dashscope_endpoint(base_url: str | None) -> bool:
    """Match DashScope / Aliyun-region endpoints.

    Blank base_url falls back to DashScope because the ``QwenConfig`` default
    points there (see ``qwen/config.py:_DEFAULT_BASE_URL``).
    """
    if not base_url:
        return True
    bu = base_url.lower()
    return "dashscope" in bu or "aliyuncs.com" in bu


def QwenAgent(
    config: QwenConfig | None = None,
    *,
    state: "AgentState | None" = None,
) -> "Agent":
    """Dispatch to the concrete OpenAI-compatible agent class for ``config.base_url``.

    - DashScope / Aliyun → ``DashScopeQwenAgent`` (adds multi-key rotation).
    - Anything else (gpt-5.5 (OpenAI-compat), openai.com, ...) → ``OpenAICompatibleAgent``.

    Returns an instance, so existing ``with QwenAgent(cfg) as parent:`` and
    ``isinstance(agent, ...)`` callers keep working — they should check
    against the new concrete classes (``DashScopeQwenAgent`` /
    ``OpenAICompatibleAgent``) if they need a class-level check.
    """
    if config is None:
        config = QwenConfig()
    if _is_dashscope_endpoint(config.base_url):
        return DashScopeQwenAgent(config, state=state)
    return OpenAICompatibleAgent(config, state=state)


__all__ = [
    "DashScopeQwenAgent",
    "OpenAICompatibleAgent",
    "QwenAgent",
    "QwenConfig",
]
