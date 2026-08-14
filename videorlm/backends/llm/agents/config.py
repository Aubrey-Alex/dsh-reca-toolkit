"""Provider-neutral agent configuration.

Pre-refactor the only config was ``QwenConfig`` (DashScope-flavored), used
as the universal config for both ``OpenAICompatibleAgent`` (base) and
``DashScopeQwenAgent``, plus ``OpenAIMessagesAgent``. That conflated:

  - dashscope-specific fields (``enable_thinking``) with generic ones
  - dashscope-specific defaults (``base_url``, ``api_key`` env walk) with
    a class actually intended to back any OpenAI-compatible endpoint

After 2026-05-16 there is a three-tier hierarchy:

  ``AgentConfig``                 — base, provider-neutral (this file)
  ├─ ``QwenConfig``               — adds ``enable_thinking`` (qwen-3.x)
  └─ ``OpenAIMessagesConfig``     — adds ``thinking_budget_tokens`` (Anthropic Messages)

``api_key`` / ``base_url`` default to empty strings. The agent's
``start()`` resolves them lazily from the bound provider
(``get_provider(PROVIDER_NAME)``) when the config doesn't pin them —
single source of truth for env lookup is ``providers/<name>.py``.

``provider`` field (when non-empty) overrides the agent class's
``PROVIDER_NAME`` attribute, letting one config drive any provider.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Any


def _default_thread_id_factory() -> str:
    """Pure-Python session label for tracing/logging. The underlying
    server is stateless (full message list sent each turn), so this is
    purely informational."""
    return f"agent-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class AgentConfig:
    """Provider-neutral, immutable config for any
    ``OpenAICompatibleAgent`` subclass.

    Defaults are intentionally minimal. ``model`` is required (no
    DashScope-specific assumption); ``api_key`` / ``base_url`` resolve
    via the bound provider when empty.
    """

    # Core endpoint metadata
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    # When non-empty, overrides the agent's ``PROVIDER_NAME`` class attr
    # to pick which provider's KeyPool + classifier ``_chat_create``
    # routes through. Leave empty to inherit from the agent class.
    provider: str = ""

    # Conversation tuning
    system_prompt: str | None = None
    temperature: float = 0.5
    max_tokens: int | None = None

    # Transport
    request_timeout_s: float = 180.0
    # SDK's silent built-in retries rebind to the same api_key, fighting
    # KeyPool rotation — keep at 0 so the framework's
    # ``retry_until_exhausted`` is the only retry layer.
    sdk_max_retries: int = 0

    # Role-based pool key. When non-empty, the agent shares a slot with
    # every other agent declaring the same role — regardless of model.
    # Canonical roles live in ``framework._common.pools``:
    #   "planner" / "anchor_validator" / "segment_validator" / "render"
    # Empty role ⇒ slot is keyed by model name (legacy fallback used by
    # ad-hoc smoke scripts that don't care about role-pooling).
    role: str = ""
    # Per-pool concurrency cap. Role-aware callers pass
    # ``pools.pool_size_for_role(role)`` so every member of the same
    # role agrees on the cap (first acquirer wins the slot registry, so
    # disagreement is a silent footgun). When role="" defaults to 8.
    max_concurrency: int = 8

    # When True, ``prompt_with_images`` fetches each http(s) image URL
    # from THIS pod and inlines it as a base64 ``data:`` URI in the
    # multimodal content block, so the upstream gateway never has to
    # reach out to a third-party host. Required for OpenAI-side gateways
    # (e.g. gateway → ChatGPT/Codex) which fetch URLs from US-side
    # workers that cannot reach intra-region Aliyun OSS endpoints like
    # ``intern-data-wlcb`` and return ``Timeout while downloading``.
    # Stay False for DashScope (qwen3-vl-plus): same Aliyun region, OSS
    # URL fetch is free.
    inline_images: bool = False

    # Server-side video frame sampling rate (frames per second) for
    # ``prompt_with_video``. DashScope qwen3-vl / qwen-vl-max accept
    # [0.1, 10] in OpenAI-compat mode (``max_frames`` is native-SDK only).
    # Default 10 (max) — empirically required to reliably catch sub-second
    # artifacts (e.g. a 0.4s portrait-leak frame mid-clip; fps=2 missed
    # it 0/3 times, fps=10 hit it 2/2 with an artifact-aware prompt). The
    # fps is placed at the content-item top level (sibling of
    # ``video_url``); nesting it inside ``video_url`` is silently ignored.
    video_sample_fps: int = 10

    def with_overrides(self, **kwargs: Any) -> "AgentConfig":
        return replace(self, **kwargs)


__all__ = ["AgentConfig", "_default_thread_id_factory"]
