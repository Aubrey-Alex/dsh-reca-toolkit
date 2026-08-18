"""Small stateful client for OpenAI Responses-compatible gateways.

This adapter intentionally mirrors the existing Agent surface. It is used by
the visual validator when the configured gateway exposes ``/responses``
instead of ``/chat/completions``. Planner, retry, and repair logic stays in
the framework; this module only translates messages and parses the response.
"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from ..base import (
    Agent,
    AgentCapabilities,
    AgentError,
    AgentMessage,
    AgentState,
    append_messages,
)


def _text_content(text: str, *, role: str = "user") -> list[dict[str, str]]:
    # Responses input items use input_text for user/system turns, while a
    # replayed assistant turn must use output_text (input_text is rejected by
    # the API as an assistant content type).
    content_type = "output_text" if role == "assistant" else "input_text"
    return [{"type": content_type, "text": text}]


@contextmanager
def _global_request_lock():
    """Serialize Responses calls across ReCA worker processes.

    The Gateway can launch several ReCA runs at once, so an in-process
    semaphore is not enough for a single upstream GPT quota.  ``flock`` is
    advisory and automatically released if a worker dies.
    """
    lock_path = os.environ.get("RECA_GPT_GLOBAL_LOCK_PATH", "/tmp/reca-gpt-responses.lock")
    try:
        import fcntl
    except ImportError:
        # Keep the client usable on platforms without POSIX file locking.
        yield
        return

    handle = None
    try:
        path = Path(lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        if handle is not None:
            handle.close()
        yield
        return

    try:
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class OpenAIResponsesAgent(Agent):
    """Stateless HTTP Responses API agent with forkable local history."""

    AGENT_NAME = "openai_responses"
    FAMILY = "openai-responses"

    def __init__(self, config: Any, *, state: AgentState | None = None) -> None:
        super().__init__(state=state)
        self._config = config
        self._closed = False

    @property
    def config(self) -> Any:
        return self._config

    def capabilities(self) -> AgentCapabilities:
        role = (getattr(self._config, "role", "") or "").strip()
        slot_key = f"role:{role}" if role else getattr(self._config, "model", "responses")
        return AgentCapabilities(
            agent_name=slot_key,
            family=self.FAMILY,
            max_concurrency=int(getattr(self._config, "max_concurrency", 8) or 8),
            notes="OpenAI Responses API over HTTP",
        )

    def start(self) -> None:
        if self._closed:
            raise AgentError("agent has been closed; create a new one")
        if not self._state.get("messages"):
            system_prompt = getattr(self._config, "system_prompt", None)
            if system_prompt:
                append_messages(self._state, AgentMessage(role="system", content=system_prompt))

    def close(self) -> None:
        self._closed = True

    def load_session(self, thread_id: str) -> None:
        self._state["thread_id"] = thread_id

    def _endpoint(self) -> str:
        endpoint = str(getattr(self._config, "base_url", "") or "").rstrip("/")
        if not endpoint:
            raise AgentError("OpenAIResponsesAgent requires base_url")
        if not endpoint.endswith("/responses"):
            endpoint += "/responses"
        return endpoint

    def _input_messages(self, current: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in self._state.get("messages", []) or []:
            payload.append({
                "role": message.role,
                "content": _text_content(message.content, role=message.role),
            })
        if current is not None:
            payload.append({"role": "user", "content": current})
        return payload

    def _request(self, input_messages: list[dict[str, Any]]) -> str:
        api_key = str(getattr(self._config, "api_key", "") or "")
        if not api_key:
            raise AgentError("OpenAIResponsesAgent requires api_key")
        body: dict[str, Any] = {
            "model": getattr(self._config, "model", ""),
            "input": input_messages,
            "stream": False,
        }
        max_tokens = getattr(self._config, "max_tokens", None)
        if max_tokens is not None:
            body["max_output_tokens"] = int(max_tokens)
        service_tier = os.environ.get("RECA_GPT_SERVICE_TIER", "").strip()
        if service_tier:
            body["service_tier"] = service_tier

        timeout_s = float(getattr(self._config, "request_timeout_s", 900.0) or 900.0)
        timeout = httpx.Timeout(timeout_s, connect=min(30.0, timeout_s))
        retry_statuses = {408, 429, 500, 502, 503, 504}
        try:
            max_retries = max(0, int(os.environ.get("RECA_GPT_MAX_RETRIES", "3")))
        except ValueError:
            max_retries = 3
        try:
            retry_base_s = max(0.1, float(os.environ.get("RECA_GPT_RETRY_BASE_S", "2")))
        except ValueError:
            retry_base_s = 2.0

        response = None
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                # Lock each upstream attempt, including requests from other
                # Gateway child processes, so one key is never burst-loaded.
                with _global_request_lock():
                    response = httpx.post(
                        self._endpoint(),
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=timeout,
                    )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                response = None
                last_error = exc
                if attempt >= max_retries:
                    raise AgentError(
                        f"Responses API request failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            else:
                if response.status_code < 400:
                    break
                if response.status_code not in retry_statuses or attempt >= max_retries:
                    raise AgentError(
                        f"Responses API HTTP {response.status_code} after {attempt + 1} attempts: "
                        f"{response.text[:1000]}"
                    )
                last_error = AgentError(f"HTTP {response.status_code}")
            # Jitter prevents several independently resumed runs from
            # retrying the upstream at the exact same instant.
            time.sleep(retry_base_s * (2**attempt) + random.uniform(0, 0.5))
        if response is None:
            raise AgentError(f"Responses API request failed: {last_error}") from last_error
        try:
            data = response.json()
        except ValueError as exc:
            raise AgentError("Responses API returned non-JSON data") from exc
        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text
        parts: list[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                # Routify/OpenAI Responses variants use either `text` or
                # `output_text` on message content items. Some also wrap the
                # value in an object; accept all three without treating a
                # completed response as an empty-output failure.
                value = content.get("text") or content.get("output_text")
                if isinstance(value, dict):
                    value = value.get("text") or value.get("value")
                if isinstance(value, str):
                    parts.append(value)
        result = "".join(parts)
        if not result.strip():
            raise AgentError("Responses API returned no output text")
        return result

    def _complete(self, user_text: str, content: list[dict[str, Any]] | None = None) -> str:
        if self._closed:
            raise AgentError("agent is closed")
        current = content or _text_content(user_text)
        reply = self._request(self._input_messages(current))
        append_messages(
            self._state,
            AgentMessage(role="user", content=user_text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    def _prompt(self, text: str) -> str:
        # Agent.prompt() owns the per-role semaphore. Acquiring it again here
        # deadlocks when max_concurrency workers call the public method.
        return self._complete(text)

    def _prompt_with_images(self, text: str, image_urls: list[str]) -> str:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        for url in image_urls:
            if url:
                if getattr(self._config, "inline_images", False):
                    from ..openai_compat.agent import _to_inline_data_uri
                    url = _to_inline_data_uri(url)
                content.append({"type": "input_image", "image_url": url})
        # Agent.prompt_with_images() already owns the per-role semaphore.
        return self._complete(text, content)

    def _fork_impl(self, cloned_state: AgentState) -> "OpenAIResponsesAgent":
        forked = type(self)(self._config, state=cloned_state)
        forked.start()
        return forked


__all__ = ["OpenAIResponsesAgent"]
