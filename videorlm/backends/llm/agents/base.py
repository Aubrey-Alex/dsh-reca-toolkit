"""Agent abstractions shared by every backend in this subpackage.

Self-contained on purpose: no imports from the rest of `backends/` or `project/`.
Each agent is a stateful conversational handle with the same five public methods —
`start` / `close` / `prompt` / `load_session` / `fork` — and declares a
`max_concurrency` so the base class can enforce a per-agent-name BoundedSemaphore
around `prompt()` without callers having to wire up their own throttle.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class AgentMessage:
    """Single conversational turn stored in `AgentState["messages"]`."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"AgentMessage.role={self.role!r} invalid")
        if self.content is None:
            raise ValueError("AgentMessage.content is None")


class AgentState(TypedDict, total=False):
    """Conversation state owned by an `Agent`.

    `messages` is the only required field; `thread_id` is set by `start()` /
    `load_session()` once the underlying session has an id (codex-acp SessionId,
    pure-Python label for stateless backends, etc.).
    """

    messages: list[AgentMessage]
    thread_id: NotRequired[str]


@dataclass(frozen=True)
class AgentCapabilities:
    """Static metadata used for routing and central concurrency throttling."""

    agent_name: str
    family: str
    max_concurrency: int
    requests_per_minute: int = 0
    notes: str = ""


class AgentError(RuntimeError):
    """Raised on lifecycle / transport errors inside an Agent."""


# --- per-agent-name concurrency registry -----------------------------------
#
# Backed by ``NamedSemaphorePool`` (``_common/concurrency.py``) — same class
# powers ``media/interface/dispatch.py``'s per-backend dispatch semaphore.
# No env-prefix here: agent caps come from ``AgentCapabilities.max_concurrency``
# unconditionally. (Use ``RECA_LLM_CONCURRENCY_*``-style overrides at the caps
# construction site if you ever need them.)

from ..._common.concurrency import NamedSemaphorePool as _NamedSemaphorePool


_AGENT_SLOTS = _NamedSemaphorePool()


def slot(agent_name: str, max_concurrency: int) -> "Iterator[None]":
    """Acquire a slot from the agent-name semaphore for the duration of a block.

    Used internally by `Agent.prompt()`; exposed so callers that want to throttle
    a non-prompt code path (e.g. parallel `fork()`s) can reuse the same lattice.
    """
    return _AGENT_SLOTS.slot(agent_name, max_concurrency)


def _reset_semaphores_for_tests() -> None:
    _AGENT_SLOTS.reset_for_tests()


# --- Agent ABC --------------------------------------------------------------


class Agent(ABC):
    """Stateful conversational handle. Concrete agents implement `_prompt()`.

    Public surface (5 methods):
        start, close, prompt, load_session, fork

    `prompt()` is a template method: it acquires a per-agent-name slot via the
    capabilities-declared `max_concurrency`, then calls `_prompt()` for the real
    work. Subclasses do not override `prompt()`.
    """

    def __init__(self, *, state: AgentState | None = None) -> None:
        if state is None:
            self._state: AgentState = {"messages": []}
        else:
            state.setdefault("messages", [])
            self._state = state
        self._closed = False

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def thread_id(self) -> str | None:
        return self._state.get("thread_id")

    @property
    def closed(self) -> bool:
        return self._closed

    @abstractmethod
    def capabilities(self) -> AgentCapabilities:
        """Return static capability metadata."""

    @abstractmethod
    def start(self) -> None:
        """Open any underlying session/process. Idempotent on already-started."""

    @abstractmethod
    def close(self) -> None:
        """Shut everything down. Idempotent."""

    @abstractmethod
    def load_session(self, thread_id: str) -> None:
        """Bind this agent to an existing session id (provider-specific)."""

    @abstractmethod
    def _prompt(self, text: str) -> str:
        """Provider-specific implementation of one user turn.

        Must append the user `AgentMessage` and the resulting assistant
        `AgentMessage` to `self._state["messages"]` and return the assistant text.
        Called inside the concurrency slot.
        """

    @abstractmethod
    def _fork_impl(self, cloned_state: AgentState) -> Agent:
        """Provider-specific fork: build a new agent on the cloned state.

        Called by `fork()` after the state has been deep-copied.
        """

    def prompt(self, text: str) -> str:
        if self._closed:
            raise AgentError("agent is closed")
        cap = self.capabilities()
        with slot(cap.agent_name, cap.max_concurrency):
            return self._prompt(text)

    def prompt_with_images(self, text: str, image_urls: list[str]) -> str:
        """Multimodal prompt: text + image URLs.

        Each subclass must implement `_prompt_with_images`. The default
        raises ``AgentError`` so callers that pass images to a text-only
        backend get a clear error rather than a silent fallback.

        Args:
            text: user prompt text.
            image_urls: list of public/OSS image URLs (no local paths).
        """
        if self._closed:
            raise AgentError("agent is closed")
        if not image_urls:
            return self.prompt(text)
        cap = self.capabilities()
        with slot(cap.agent_name, cap.max_concurrency):
            return self._prompt_with_images(text, image_urls)

    def _prompt_with_images(self, text: str, image_urls: list[str]) -> str:
        raise AgentError(
            f"{type(self).__name__}: _prompt_with_images not implemented "
            f"(text-only backend). Use a vision-capable agent / model."
        )

    def prompt_with_video(
        self,
        text: str,
        video_url: str,
        image_urls: list[str] | None = None,
    ) -> str:
        """Multimodal prompt: text + ONE video URL + optional image_url list.

        Used by the segment validator: the video is what's being judged
        (model samples its own frames over the timeline), the image_urls
        are ref/identity references (portraits).

        Args:
            text: user prompt text.
            video_url: public/OSS URL of the .mp4 to judge.
            image_urls: optional list of additional image_url items
                (portraits, anchors). Pass [] / None for video-only.
        """
        if self._closed:
            raise AgentError("agent is closed")
        if not video_url:
            # Degrade to image-only or text-only if no video supplied.
            return self.prompt_with_images(text, image_urls or [])
        cap = self.capabilities()
        with slot(cap.agent_name, cap.max_concurrency):
            return self._prompt_with_video(text, video_url, image_urls or [])

    def _prompt_with_video(
        self,
        text: str,
        video_url: str,
        image_urls: list[str],
    ) -> str:
        raise AgentError(
            f"{type(self).__name__}: _prompt_with_video not implemented "
            f"(text-only or image-only backend). Use a vision-capable agent / model."
        )

    def fork(self) -> Agent:
        if self._closed:
            raise AgentError("agent is closed; cannot fork")
        cloned = _deepcopy_state(self._state)
        return self._fork_impl(cloned)

    def __enter__(self) -> Agent:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --- helpers ----------------------------------------------------------------


def _deepcopy_state(state: AgentState) -> AgentState:
    """Deep-copy state. `AgentMessage` is frozen, so `copy.deepcopy` is safe."""
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if key == "messages":
            cloned[key] = [copy.deepcopy(m) for m in value or []]
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned  # type: ignore[return-value]


def append_messages(state: AgentState, *messages: AgentMessage) -> None:
    """Append AgentMessages to state.messages, materialising the list lazily."""
    existing = list(state.get("messages") or [])
    existing.extend(messages)
    state["messages"] = existing


__all__ = [
    "Agent",
    "AgentCapabilities",
    "AgentError",
    "AgentMessage",
    "AgentState",
    "Role",
    "append_messages",
    "slot",
]


# Internal helper for tests; not part of the public API.
__test_helpers__ = {"_reset_semaphores_for_tests": _reset_semaphores_for_tests}
