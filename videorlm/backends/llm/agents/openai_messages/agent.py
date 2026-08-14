"""OpenAIMessagesAgent — Anthropic Messages API ingress over an OpenAI-compatible gateway.

Talks directly to ``${base_url}/v1/messages`` with stream=True and parses
the Anthropic SSE event stream by hand (no SDK). We accumulate
``content_block_delta`` deltas whose ``type=="text_delta"`` into the
assistant reply; ``thinking_delta`` and other events are ignored.

State machinery (``messages`` list, system seed, fork via deep-copy,
agent slot acquisition) mirrors ``OpenAICompatibleAgent`` so this drops
into anywhere a QwenAgent / OpenAICompatibleAgent is wired.
"""
from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import httpx

from ..base import (
    Agent,
    AgentCapabilities,
    AgentError,
    AgentMessage,
    AgentState,
    append_messages,
)

if TYPE_CHECKING:
    from .config import OpenAIMessagesConfig


class OpenAIMessagesAgent(Agent):
    """Anthropic Messages API agent on an OpenAI-compatible gateway.

    Same five public methods as the base Agent. Uses
    ``OpenAIMessagesConfig`` (extends ``AgentConfig`` with
    ``thinking_budget_tokens``) — pre-2026-05-16 it borrowed
    ``QwenConfig`` and the Anthropic-only thinking budget field lived
    there, conflating dashscope and anthropic concerns.
    """

    AGENT_NAME = "openai_messages"
    FAMILY = "anthropic-messages-text"
    PROVIDER_NAME = "openai"   # Anthropic Messages over OpenAI-compat gateway

    def __init__(
        self,
        config: "OpenAIMessagesConfig | None" = None,
        *,
        state: AgentState | None = None,
    ) -> None:
        super().__init__(state=state)
        if config is None:
            from .config import OpenAIMessagesConfig as _Cfg
            config = _Cfg()
        self._config = config
        self._client: httpx.Client | None = None
        # Resolved in ``start()`` from config + provider; empty until then.
        self._api_key: str = ""
        self._base_url: str = ""

    # --- introspection ------------------------------------------------------

    @property
    def config(self) -> "OpenAIMessagesConfig":
        return self._config

    def capabilities(self) -> AgentCapabilities:
        role = (self._config.role or "").strip()
        slot_key = f"role:{role}" if role else self._config.model
        return AgentCapabilities(
            agent_name=slot_key,
            family=self.FAMILY,
            max_concurrency=self._config.max_concurrency,
            requests_per_minute=0,
            notes=(
                f"Anthropic Messages API (/v1/messages) on OpenAI-compatible gateway. "
                f"thinking_budget={self._config.thinking_budget_tokens}  "
                f"slot_key={slot_key!r} cap={self._config.max_concurrency}"
            ),
        )

    # --- lifecycle ----------------------------------------------------------

    def _resolve_url(self) -> str:
        base = (self._base_url or "").rstrip("/")
        if not base:
            raise AgentError("OpenAIMessagesAgent requires base_url")
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/messages"

    def start(self) -> None:
        if self._closed:
            raise AgentError("agent has been closed; create a new one")
        if self._client is not None:
            return
        # Resolve api_key / base_url from config first, then fall back to
        # the bound provider (lookup chain lives in
        # ``backends._common.providers.<name>``). Same resolution as
        # ``OpenAICompatibleAgent.start()``.
        from ...._common.env import env_value
        from ...._common.platforms import get_provider

        provider_name = (self._config.provider or self.PROVIDER_NAME).strip()
        try:
            prov = get_provider(provider_name)
        except KeyError as exc:
            raise AgentError(
                f"OpenAIMessagesAgent: unknown provider {provider_name!r}"
            ) from exc

        self._api_key = self._config.api_key or prov.api_key()
        if not self._api_key:
            raise AgentError(
                f"OpenAIMessagesAgent requires api_key for provider="
                f"{provider_name!r}: set one of {prov.api_key_envs!r} or "
                "pass api_key=... on the config"
            )
        self._base_url = (
            self._config.base_url
            or env_value("OPENAI_BASE_URL").strip()
            or prov.base_url
            or ""
        )
        # connect=10s, read=request_timeout_s (long for reasoning), write/pool=10s
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=max(60.0, self._config.request_timeout_s),
                write=10.0,
                pool=10.0,
            ),
            headers={"Accept-Encoding": "identity"},
        )
        if not self._state.get("thread_id"):
            self._state["thread_id"] = f"msgs-{uuid.uuid4().hex[:12]}"
        msgs = self._state.get("messages") or []
        has_system = any(m.role == "system" for m in msgs)
        if self._config.system_prompt and not has_system:
            append_messages(
                self._state,
                AgentMessage(role="system", content=self._config.system_prompt),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # --- session -----------------------------------------------------------

    def load_session(self, thread_id: str) -> None:
        if self._closed:
            raise AgentError("agent is closed")
        self._state["thread_id"] = thread_id

    # --- prompt -----------------------------------------------------------

    def _build_body(self, history: list[AgentMessage], current_user_text: str) -> dict[str, Any]:
        # Anthropic format: system is a top-level field (string OR list of blocks),
        # the messages array carries only user/assistant turns.
        system_chunks: list[str] = []
        turns: list[dict[str, Any]] = []
        for m in history:
            if m.role == "system":
                system_chunks.append(m.content)
            else:
                turns.append({"role": m.role, "content": m.content})
        turns.append({"role": "user", "content": current_user_text})

        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": turns,
            "max_tokens": self._config.max_tokens or 32000,
            "stream": True,
        }
        if system_chunks:
            body["system"] = "\n\n".join(system_chunks)
        if self._config.thinking_budget_tokens > 0:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._config.thinking_budget_tokens,
            }
        return body

    def _prompt(self, text: str) -> str:
        if self._client is None:
            raise AgentError("OpenAIMessagesAgent.prompt called before start()")
        history = list(self._state.get("messages") or [])
        body = self._build_body(history, text)
        url = self._resolve_url()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
        }

        text_parts: list[str] = []
        try:
            with self._client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    err_body = resp.read().decode(errors="replace")[:600]
                    raise AgentError(
                        f"OpenAIMessagesAgent: HTTP {resp.status_code} from {url}: {err_body}"
                    )
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line.removeprefix("data:").strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") != "content_block_delta":
                        continue
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        piece = delta.get("text") or ""
                        if piece:
                            text_parts.append(piece)
        except AgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                f"OpenAIMessagesAgent.prompt failed: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc

        reply = "".join(text_parts)
        if not reply:
            raise AgentError(
                f"OpenAIMessagesAgent: streaming returned empty text "
                f"(model={self._config.model!r})"
            )
        append_messages(
            self._state,
            AgentMessage(role="user", content=text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    # --- fork --------------------------------------------------------------

    def _fork_impl(self, cloned_state: AgentState) -> Agent:
        # Same config, fresh client (httpx.Client isn't fork-safe), new uuid.
        return OpenAIMessagesAgent(self._config, state=cloned_state)
