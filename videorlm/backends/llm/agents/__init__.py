"""Stateful, fork-able conversational agents — central concurrency-managed.

Public surface:
    Agent, AgentMessage, AgentState, AgentCapabilities, AgentError
    CodexAgent, CodexConfig                  — codex-acp Rust subprocess wrapper
    OpenAICompatibleAgent                    — generic OpenAI-compat chat (openai-compat)
    DashScopeQwenAgent                       — OpenAI-compat + multi-key rotation (DashScope)
    QwenAgent (factory), QwenConfig          — picks one of the two by ``config.base_url``

Every concrete agent has the same five-method contract: `start`, `close`,
`prompt`, `load_session`, `fork`. `prompt()` is throttled by a per-agent-name
BoundedSemaphore declared via `AgentCapabilities.max_concurrency`, so callers
get central concurrency management for free.
"""

from .base import (
    Agent,
    AgentCapabilities,
    AgentError,
    AgentMessage,
    AgentState,
    Role,
    append_messages,
    slot,
)
from .codex import CodexAgent, CodexConfig
from .openai_compat import OpenAICompatibleAgent, OpenAICompatibleConfig
from .openai_messages import OpenAIMessagesAgent
from .openai_responses import OpenAIResponsesAgent
from .qwen import DashScopeQwenAgent, QwenAgent, QwenConfig

__all__ = [
    "Agent",
    "AgentCapabilities",
    "AgentError",
    "AgentMessage",
    "AgentState",
    "CodexAgent",
    "CodexConfig",
    "DashScopeQwenAgent",
    "OpenAICompatibleAgent",
    "OpenAICompatibleConfig",
    "OpenAIMessagesAgent",
    "OpenAIResponsesAgent",
    "QwenAgent",
    "QwenConfig",
    "Role",
    "append_messages",
    "slot",
]
