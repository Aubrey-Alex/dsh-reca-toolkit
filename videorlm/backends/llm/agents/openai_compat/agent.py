"""OpenAICompatibleAgent: stateful, fork-able agent backed by any OpenAI-compatible
chat-completions endpoint (DashScope, gpt-5.5 (OpenAI-compat), openai.com, etc.).

This is the generic base. Subclasses override ``_chat_create`` to add
provider-specific dispatch (see ``DashScopeQwenAgent`` in
``..qwen.agent`` for the multi-key rotation flavor).

Conversation state lives entirely in ``AgentState["messages"]``; each
``prompt()`` POSTs the full message list, so ``fork()`` is a straight
deep-copy without server-side handoff.
"""

from __future__ import annotations

import base64
import mimetypes
import uuid
from typing import TYPE_CHECKING

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
    # Type-only import to break the circular dependency: qwen/__init__.py
    # imports OpenAICompatibleAgent from this module, so we can't pull
    # QwenConfig in eagerly at module load. The default-config path
    # imports lazily inside __init__ below.
    from ..qwen.config import QwenConfig


def _to_inline_data_uri(url: str, *, timeout_s: float = 60.0) -> str:
    """Fetch ``url`` from this pod and return a ``data:<mime>;base64,...`` URI.

    Used by ``_prompt_with_images`` / ``_prompt_with_video`` when the agent's
    config has ``inline_images=True`` (anchor-validator path on the gateway
    gateway, which can't reach Aliyun intra-region OSS hosts).

    Resolution order:
      1. ``data:`` URI → pass through unchanged.
      2. ``http(s)://`` URL → ``httpx.get``.
      3. Anything else → treated as a local filesystem path; bytes read
         directly from disk. (Anchor validator may pass a local PNG path
         when the OSS URL is missing — this avoids the prior
         ``UnsupportedProtocol`` crash that silently disabled the
         anchor-validator on every anchor.)
    """
    if url.startswith("data:"):
        return url
    if url.startswith(("http://", "https://")):
        rsp = httpx.get(url, timeout=timeout_s)
        rsp.raise_for_status()
        content = rsp.content
        mime = (rsp.headers.get("content-type") or "").split(";")[0].strip()
    else:
        import os
        if not os.path.exists(url):
            raise FileNotFoundError(
                f"_to_inline_data_uri: local path does not exist: {url}"
            )
        with open(url, "rb") as f:
            content = f.read()
        mime = ""
    if not mime or mime == "application/octet-stream":
        ext = url.rsplit(".", 1)[-1].lower() if "." in url else ""
        mime = mimetypes.types_map.get(f".{ext}", "image/png")
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _raise_if_stream_error(chunk: object, *, label: str) -> None:
    """Surface upstream error events the openai SDK doesn't auto-raise.

    DashScope (and some OpenAI-compat gateways) occasionally embed an
    ``{"error": {...}}`` envelope as a regular SSE chunk inside an HTTP
    200 response. The openai SDK parses it into a ``ChatCompletionChunk``
    pydantic model whose ``choices`` is empty; the error payload lands
    in ``model_extra``. Without this check, the iteration loop silently
    drops the chunk and we end up with an empty reply masquerading as a
    successful call.
    """
    err = getattr(chunk, "error", None)
    if err is None:
        extra = getattr(chunk, "model_extra", None) or {}
        if isinstance(extra, dict):
            err = extra.get("error")
    if err:
        raise AgentError(f"{label}: upstream stream error: {str(err)[:600]}")


class OpenAICompatibleAgent(Agent):
    """Generic OpenAI-compatible chat agent. Stateless server-side, full message
    list resent on every turn. ``fork()`` = pure deep-copy.

    ``PROVIDER_NAME`` selects which provider's KeyPool + classifier
    ``_chat_create`` funnels through. Subclasses (e.g. ``DashScopeQwenAgent``)
    override it to ``"dashscope"`` so they share the dashscope key pool +
    cooldown + EWMA with every other dashscope caller. Anything else
    (openai.com, gpt-5.5 (OpenAI-compat), self-hosted) keeps the default
    ``"openai"`` and lets ``OPENAI_BASE_URL`` env do the routing.
    """

    AGENT_NAME = "openai_compat"
    FAMILY = "openai-compat-text"
    PROVIDER_NAME = "openai"

    def __init__(
        self,
        config: "QwenConfig | None" = None,
        *,
        state: AgentState | None = None,
    ) -> None:
        super().__init__(state=state)
        if config is None:
            # Lazy import — see TYPE_CHECKING note above.
            from ..qwen.config import QwenConfig as _QwenConfig
            config = _QwenConfig()
        self._config = config
        self._client: object | None = None  # lazy openai.OpenAI

    # --- introspection ------------------------------------------------------

    @property
    def config(self) -> QwenConfig:
        return self._config

    def capabilities(self) -> AgentCapabilities:
        # Slot key = role when set (so all agents in the same role share
        # one pool regardless of model), else fall back to per-model.
        role = (self._config.role or "").strip()
        slot_key = f"role:{role}" if role else self._config.model
        return AgentCapabilities(
            agent_name=slot_key,
            family=self.FAMILY,
            max_concurrency=self._config.max_concurrency,
            requests_per_minute=0,
            notes=(
                f"OpenAI-compatible HTTP endpoint, pure-HTTP. "
                f"Server is stateless; full message list resent each turn. "
                f"slot_key={slot_key!r} cap={self._config.max_concurrency}"
            ),
        )

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._closed:
            raise AgentError("agent has been closed; create a new one")
        if self._client is not None:
            return
        # Resolve provider, api_key, base_url. Order:
        #   1. ``config.provider`` if set, else this class's ``PROVIDER_NAME``.
        #   2. ``config.api_key`` if set, else ``provider.api_key()``.
        #   3. ``config.base_url`` if set, else (only for provider="openai")
        #      ``OPENAI_BASE_URL`` env, else
        #      ``provider.base_url``.
        # The OPENAI_BASE_URL env walk is **scoped to provider="openai"** on
        # purpose: prior to 2026-05-17 it was unconditional, which meant
        # any DashScopeQwenAgent that didn't explicitly pin base_url got
        # routed to the OpenAI-compatible gateway (because our .env sets
        # OPENAI_BASE_URL=... for gpt-image-2
        # / gpt-5.5). The validator path hit this exact trap and POSTed
        # qwen3-vl-plus calls to gateway's index page. Provider-scoped env
        # makes the agent's declared PROVIDER_NAME authoritative again.
        from ...._common.env import env_value
        from ...._common.platforms import get_provider

        provider_name = (self._config.provider or self.PROVIDER_NAME).strip()
        try:
            prov = get_provider(provider_name)
        except KeyError as exc:
            raise AgentError(
                f"{type(self).__name__}: unknown provider {provider_name!r}; "
                f"set config.provider or PROVIDER_NAME to a registered one"
            ) from exc

        api_key = self._config.api_key or prov.api_key()
        if not api_key:
            raise AgentError(
                f"{type(self).__name__} requires api_key for provider="
                f"{provider_name!r}: set one of {prov.api_key_envs!r} or pass "
                "api_key=... on the config"
            )
        base_url = self._config.base_url
        if not base_url and provider_name == "openai":
            # OpenAI gateway routing (OpenAI-compat, openai.com proper, anthropic
            # bridges) is the only provider whose endpoint is routinely
            # swapped via env. Other providers (dashscope, kling, etc.)
            # always go to their fixed prov.base_url unless the config
            # pinned a different one.
            base_url = env_value("OPENAI_BASE_URL").strip()
        if not base_url:
            base_url = prov.base_url or None   # OpenAI SDK falls back to api.openai.com

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentError(
                f"{type(self).__name__} needs the `openai` package: pip install openai"
            ) from exc
        sdk_timeout = max(30.0, self._config.request_timeout_s - 10.0)
        # 强制 identity 编码: gpt-5.5 (OpenAI-compat) endpoint 经 Cloudflare 中转, CF 对
        # 压缩的 SSE 流做缓冲, 长 stream (>120-300s 或 >5000 chars) 会被边缘节点
        # 切掉 (silent truncation 或 RemoteProtocolError). 实测同 prompt:
        # `Accept-Encoding: gzip,deflate,br,zstd` 切在 4712 chars / 123s;
        # `Accept-Encoding: identity` 完整跑完 1.6MB / 181s. DashScope 不经 CF,
        # 加这个 header 也无副作用. 见 2026-05-15 diag in commit history.
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=sdk_timeout,
            max_retries=self._config.sdk_max_retries,
            default_headers={"Accept-Encoding": "identity"},
        )
        if not self._state.get("thread_id"):
            self._state["thread_id"] = f"agent-{uuid.uuid4().hex[:12]}"
        # Seed system prompt into messages on first start, if configured and not
        # already present from a restored state.
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
        # OpenAI client has no explicit close in the sync flavour; drop the ref.
        self._client = None

    # --- provider dispatch hook --------------------------------------------

    def _chat_create(self, client: object, **kwargs):
        """Funnel the SDK call through ``with_key(PROVIDER_NAME, ...)`` so
        every agent — base or subclass — gets multi-key rotation +
        cooldown + EWMA from the shared KeyPool.

        Falls through to the raw ``client.chat.completions.create`` when
        no keys are configured for ``PROVIDER_NAME`` (dev / local mode
        where ``QwenConfig.api_key`` is a single pinned key)."""
        from ...._common.platforms import get_provider, with_key

        try:
            pool = get_provider(self.PROVIDER_NAME).key_pool()
        except Exception:  # noqa: BLE001 — missing env, unknown provider, etc.
            return client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        if not pool.keys:
            return client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        return with_key(
            self.PROVIDER_NAME,
            lambda api_key: client.with_options(  # type: ignore[union-attr]
                api_key=api_key,
            ).chat.completions.create(**kwargs),
        )

    # --- session management -------------------------------------------------

    def load_session(self, thread_id: str) -> None:
        if self._closed:
            raise AgentError("agent is closed")
        # Stateless server-side; load_session just relabels the in-memory
        # session. Caller is expected to have populated state["messages"]
        # from whatever store they persist conversations in.
        self._state["thread_id"] = thread_id

    # --- prompt impl --------------------------------------------------------

    def _prompt(self, text: str) -> str:
        client = self._require_client()
        history = list(self._state.get("messages") or [])
        history.append(AgentMessage(role="user", content=text))
        payload = [{"role": m.role, "content": m.content} for m in history]
        kwargs: dict = {
            "model": self._config.model,
            "messages": payload,
            "temperature": self._config.temperature,
            # Streaming dodges Cloudflare 524 (origin >120s before first byte):
            # once chunks start flowing, the proxy treats the conn as active.
            "stream": True,
        }
        if self._config.max_tokens is not None:
            kwargs["max_tokens"] = self._config.max_tokens
        if getattr(self._config, "enable_thinking", False):
            # DashScope qwen3 thinking mode. ``extra_body`` is the openai
            # SDK escape hatch for non-standard top-level fields; here it
            # surfaces as ``"enable_thinking": True`` in the JSON body.
            kwargs["extra_body"] = {"enable_thinking": True}
        try:
            stream = self._chat_create(client, **kwargs)
            chunks: list[str] = []
            for chunk in stream:
                _raise_if_stream_error(chunk, label=f"{type(self).__name__}.prompt")
                try:
                    delta = chunk.choices[0].delta
                except (AttributeError, IndexError):
                    continue
                piece = getattr(delta, "content", None) or ""
                if piece:
                    chunks.append(piece)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                f"{type(self).__name__}.prompt failed: "
                f"{type(exc).__name__}: {str(exc)[:1500]}"
            ) from exc
        reply = "".join(chunks)
        if not reply:
            raise AgentError(
                f"{type(self).__name__}: streaming returned empty content "
                f"(model={self._config.model!r})"
            )
        append_messages(
            self._state,
            AgentMessage(role="user", content=text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    def _prompt_with_images(self, text: str, image_urls: list[str]) -> str:
        """Multimodal prompt: send text + image_url content blocks.

        OpenAI-compatible multimodal message format:
          {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "https://..."}},
            ...
          ]}

        Only the **current** user turn carries images; prior history stays
        text-only (image content blocks would double-charge tokens on re-send).
        The model must be vision-capable (e.g. qwen3-vl-plus) — text-only
        models like qwen3.6-plus will return ``AgentError`` from the gateway.
        """
        client = self._require_client()
        history = list(self._state.get("messages") or [])
        # Prior history: keep text-only
        payload: list[dict] = [
            {"role": m.role, "content": m.content} for m in history
        ]
        # Current user turn: multimodal content
        current_content: list[dict] = [{"type": "text", "text": text}]
        for url in image_urls:
            if not url:
                continue
            # If the upstream gateway can't reach our image host (e.g.
            # gateway → ChatGPT/Codex fetching `intern-data-wlcb` Aliyun
            # OSS, which times out 100% from US-side workers), the
            # backend config sets ``inline_images=True`` and we fetch
            # the bytes here and inline as base64. Adds one round-trip
            # per image from THIS pod (which CAN reach the host), but
            # the gateway then sees a self-contained data URI.
            if self._config.inline_images:
                try:
                    url = _to_inline_data_uri(url)
                except Exception as exc:  # noqa: BLE001
                    raise AgentError(
                        f"{type(self).__name__}._prompt_with_images: failed to inline image "
                        f"{url[:120]}: {type(exc).__name__}: {str(exc)[:300]}"
                    ) from exc
            current_content.append(
                {"type": "image_url", "image_url": {"url": url}}
            )
        payload.append({"role": "user", "content": current_content})

        kwargs: dict = {
            "model": self._config.model,
            "messages": payload,
            "temperature": self._config.temperature,
            "stream": True,
        }
        if self._config.max_tokens is not None:
            kwargs["max_tokens"] = self._config.max_tokens
        try:
            stream = self._chat_create(client, **kwargs)
            chunks: list[str] = []
            for chunk in stream:
                _raise_if_stream_error(chunk, label=f"{type(self).__name__}.prompt_with_images")
                try:
                    delta = chunk.choices[0].delta
                except (AttributeError, IndexError):
                    continue
                piece = getattr(delta, "content", None) or ""
                if piece:
                    chunks.append(piece)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                f"{type(self).__name__}.prompt_with_images failed "
                f"(model={self._config.model!r}, n_images={len(image_urls)}): "
                f"{type(exc).__name__}: {str(exc)[:1500]}"
            ) from exc
        reply = "".join(chunks)
        if not reply:
            raise AgentError(
                f"{type(self).__name__}.prompt_with_images: streaming returned empty content "
                f"(model={self._config.model!r}, n_images={len(image_urls)})"
            )
        # Persist text-only versions of user/assistant in state so history
        # stays compact (images are URLs we don't keep around).
        append_messages(
            self._state,
            AgentMessage(role="user", content=text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    def _prompt_with_video(
        self,
        text: str,
        video_url: str,
        image_urls: list[str],
    ) -> str:
        """OpenAI-compat video understanding via ``video_url`` content.

        Builds the same role=user message but with a ``video_url`` content
        item carrying the segment mp4 URL. qwen3-vl-plus internally samples
        frames over the video timeline — this is the ONLY way our
        validator gets to see temporal/physical-contact continuity.

        Image refs (portraits) are still passed alongside as ``image_url``
        items for identity grounding.
        """
        client = self._require_client()
        history = list(self._state.get("messages") or [])
        payload: list[dict] = [
            {"role": m.role, "content": m.content} for m in history
        ]
        current_content: list[dict] = [{"type": "text", "text": text}]
        # The video being judged comes first so the model treats it as
        # the primary visual evidence. ``fps`` is at the content-item
        # top level (sibling of ``video_url``) — DashScope silently
        # ignores it when nested inside ``video_url``.
        video_item: dict = {"type": "video_url", "video_url": {"url": video_url}}
        sample_fps = max(1, min(10, int(getattr(self._config, "video_sample_fps", 10) or 10)))
        video_item["fps"] = sample_fps
        current_content.append(video_item)
        for url in image_urls or []:
            if not url:
                continue
            # ``inline_images`` semantics intentionally do NOT apply to the
            # video URL (a 24MB mp4 in base64 would blow past the
            # multipart body limit). Inline only the image refs when
            # the upstream gateway can't fetch them.
            if self._config.inline_images:
                try:
                    url = _to_inline_data_uri(url)
                except Exception as exc:  # noqa: BLE001
                    raise AgentError(
                        f"{type(self).__name__}._prompt_with_video: failed to inline image "
                        f"{url[:120]}: {type(exc).__name__}: {str(exc)[:300]}"
                    ) from exc
            current_content.append({"type": "image_url", "image_url": {"url": url}})
        payload.append({"role": "user", "content": current_content})

        kwargs: dict = {
            "model": self._config.model,
            "messages": payload,
            "temperature": self._config.temperature,
            "stream": True,
        }
        if self._config.max_tokens is not None:
            kwargs["max_tokens"] = self._config.max_tokens
        try:
            stream = self._chat_create(client, **kwargs)
            chunks: list[str] = []
            for chunk in stream:
                _raise_if_stream_error(chunk, label="prompt_with_video")
                try:
                    delta = chunk.choices[0].delta
                except (AttributeError, IndexError):
                    continue
                piece = getattr(delta, "content", None) or ""
                if piece:
                    chunks.append(piece)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                f"{type(self).__name__}.prompt_with_video failed "
                f"(model={self._config.model!r}, video={video_url[:80]!r}, "
                f"n_image_refs={len(image_urls or [])}): "
                f"{type(exc).__name__}: {str(exc)[:1500]}"
            ) from exc
        reply = "".join(chunks)
        if not reply:
            raise AgentError(
                f"{type(self).__name__}.prompt_with_video: streaming returned "
                f"empty content (model={self._config.model!r}, "
                f"video={video_url[:80]!r})"
            )
        append_messages(
            self._state,
            AgentMessage(role="user", content=text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    def _fork_impl(self, cloned_state: AgentState) -> Agent:
        # type(self) preserves subclass identity — DashScopeQwenAgent forks
        # to DashScopeQwenAgent (with key-pool _chat_create), not to the base.
        forked = type(self)(self._config, state=cloned_state)
        forked.start()
        return forked

    # --- internals ----------------------------------------------------------

    def _require_client(self) -> object:
        if self._closed:
            raise AgentError("agent is closed")
        if self._client is None:
            raise AgentError("agent not started; call start() first")
        return self._client
