"""CodexAgent: stateful, fork-able wrapper around the codex-acp Rust subprocess."""

from __future__ import annotations

from ..base import (
    Agent,
    AgentCapabilities,
    AgentError,
    AgentMessage,
    AgentState,
    append_messages,
)
from .acp import AcpClient
from .config import CodexConfig


class CodexAgent(Agent):
    """Codex-backed agent: one ACP child process, one ACP session.

    Implements the five-method `Agent` contract. `_prompt()` runs inside the
    base class' per-agent-name concurrency slot, so callers cannot accidentally
    overwhelm the codex-acp child by spawning N agents and prompting in parallel.
    """

    AGENT_NAME = "codex"
    FAMILY = "codex-acp"
    # Each codex-acp subprocess holds one OpenAI session; doc-strict default
    # mirrors the user-side rate limit. Override via `CodexConfig.max_concurrency`.
    DEFAULT_MAX_CONCURRENCY = 8

    def __init__(
        self,
        config: CodexConfig | None = None,
        *,
        state: AgentState | None = None,
    ) -> None:
        super().__init__(state=state)
        self._config = config if config is not None else CodexConfig()
        self._client: AcpClient | None = None

    # --- introspection ------------------------------------------------------

    @property
    def config(self) -> CodexConfig:
        return self._config

    def capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            agent_name=self.AGENT_NAME,
            family=self.FAMILY,
            max_concurrency=getattr(self._config, "max_concurrency", self.DEFAULT_MAX_CONCURRENCY),
            requests_per_minute=0,
            notes="codex-acp Rust subprocess, ACP over NDJSON stdio.",
        )

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._closed:
            raise AgentError("agent has been closed; create a new one")
        if self._client is not None:
            return
        client = AcpClient(
            binary_path=self._config.binary_path,
            cwd=self._config.cwd,
            env=self._config.env,
            request_timeout_s=self._config.request_timeout_s,
        )
        client.start()
        self._client = client
        existing = self._state.get("thread_id")
        if existing:
            sid = client.load_session(existing, self._config.cwd)
        else:
            sid = client.new_session(self._config.cwd)
            # System prompt is seeded only for fresh sessions. Resumed threads
            # already carry it via codex's persisted thread file. We also skip
            # if state.messages already has a system entry (fork carries it).
            already_seeded = any(
                m.role == "system" for m in self._state.get("messages") or []
            )
            if self._config.system_prompt and not already_seeded:
                self._seed_system_prompt(client, self._config.system_prompt)
        self._state["thread_id"] = sid

    def _seed_system_prompt(self, client: AcpClient, prompt: str) -> None:
        # Send the instructions as a hidden first ACP turn; drain the streamed
        # acknowledgement without recording the assistant chunk in state. The
        # system message itself IS recorded so callers can see / serialise it.
        for _evt in client.prompt(prompt):
            pass
        append_messages(self._state, AgentMessage(role="system", content=prompt))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    # --- session management -------------------------------------------------

    def load_session(self, thread_id: str) -> None:
        client = self._require_client()
        sid = client.load_session(thread_id, self._config.cwd)
        self._state["thread_id"] = sid

    # --- prompt impl --------------------------------------------------------

    def _prompt(self, text: str) -> str:
        client = self._require_client()
        chunks: list[str] = []
        for evt in client.prompt(text):
            if not isinstance(evt, dict):
                continue
            if evt.get("_type") == "stop":
                continue
            update = evt.get("update")
            if not isinstance(update, dict):
                continue
            if update.get("sessionUpdate") != "agent_message_chunk":
                continue
            content = update.get("content")
            if isinstance(content, dict) and content.get("type") == "text":
                part = content.get("text")
                if isinstance(part, str):
                    chunks.append(part)
        reply = "".join(chunks)
        append_messages(
            self._state,
            AgentMessage(role="user", content=text),
            AgentMessage(role="assistant", content=reply),
        )
        return reply

    def _fork_impl(self, cloned_state: AgentState) -> Agent:
        forked = CodexAgent(self._config, state=cloned_state)
        forked.start()  # sees thread_id in cloned_state, so does session/load
        return forked

    # --- internals ----------------------------------------------------------

    def _require_client(self) -> AcpClient:
        if self._closed:
            raise AgentError("agent is closed")
        if self._client is None:
            raise AgentError("agent not started; call start() first")
        return self._client
