"""Minimal ACP (Agent Client Protocol) client for the codex-acp Rust binary.

Speaks JSON-RPC 2.0 framed as newline-delimited JSON over stdio, which is what the
`agent-client-protocol` Rust crate (used by codex-acp) expects.

Method names mirror the Rust handlers in `codex-acp/src/codex_agent.rs`:
`initialize`, `session/new`, `session/load`, `session/close`, `session/prompt`.
Streamed turn output arrives as `session/update` notifications and is exposed by
`prompt()` as an iterator of update payloads.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


class AcpError(RuntimeError):
    """Raised on protocol or transport errors with the codex-acp child."""


_SENTINEL = object()


class AcpClient:
    """Thin sync wrapper around a codex-acp subprocess.

    Lifecycle: `start()` spawns the child and runs the ACP `initialize` handshake;
    `new_session()` or `load_session()` opens a session and stores its id; `prompt()`
    sends a user turn and yields streamed update payloads; `close()` shuts everything
    down. All methods are safe to call from a single owning thread.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        cwd: Path,
        env: Mapping[str, str],
        request_timeout_s: float = 600.0,
    ) -> None:
        self._binary_path = binary_path
        self._cwd = cwd
        self._env = dict(env)
        self._request_timeout_s = request_timeout_s

        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._update_queue: queue.Queue[Any] = queue.Queue()
        self._session_id: str | None = None
        self._closed = False

    @property
    def session_id(self) -> str | None:
        return self._session_id

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            return
        proc_env = {**os.environ, **self._env}
        self._proc = subprocess.Popen(
            [self._binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            cwd=str(self._cwd),
            bufsize=0,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="acp-reader", daemon=True
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_drain, name="acp-stderr", daemon=True
        )
        self._stderr_thread.start()
        self._request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.close_session()
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None

    # --- session methods ----------------------------------------------------

    def new_session(self, cwd: Path) -> str:
        result = self._request(
            "session/new",
            {"cwd": str(cwd.resolve()), "mcpServers": []},
        )
        sid = _coerce_session_id(result)
        self._session_id = sid
        return sid

    def load_session(self, thread_id: str, cwd: Path) -> str:
        result = self._request(
            "session/load",
            {
                "sessionId": thread_id,
                "cwd": str(cwd.resolve()),
                "mcpServers": [],
            },
        )
        sid = _coerce_session_id(result) or thread_id
        self._session_id = sid
        return sid

    def close_session(self) -> None:
        sid = self._session_id
        if sid is None or self._proc is None:
            self._session_id = None
            return
        try:
            self._request("session/close", {"sessionId": sid})
        except AcpError:
            pass
        self._session_id = None

    def prompt(self, text: str) -> Iterator[dict[str, Any]]:
        sid = self._session_id
        if sid is None:
            raise AcpError("no active session; call new_session/load_session first")
        params = {
            "sessionId": sid,
            "prompt": [{"type": "text", "text": text}],
        }
        # Drain any stale updates so this call's stream is clean.
        self._drain_updates()

        response_holder: dict[str, Any] = {}

        def runner() -> None:
            try:
                response_holder["result"] = self._request("session/prompt", params)
            except BaseException as exc:  # noqa: BLE001
                response_holder["error"] = exc
            finally:
                self._update_queue.put(_SENTINEL)

        sender = threading.Thread(target=runner, name="acp-prompt-sender", daemon=True)
        sender.start()

        try:
            while True:
                evt = self._update_queue.get()
                if evt is _SENTINEL:
                    break
                if isinstance(evt, dict):
                    yield evt
        finally:
            sender.join()

        err = response_holder.get("error")
        if err is not None:
            if isinstance(err, BaseException):
                raise err
            raise AcpError(str(err))
        result = response_holder.get("result")
        if isinstance(result, dict):
            yield {"_type": "stop", "result": result}

    # --- internals ----------------------------------------------------------

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise AcpError("acp client not started")
        rid = uuid.uuid4().hex
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = q
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise AcpError(f"broken pipe sending {method}") from exc
        try:
            response = q.get(timeout=self._request_timeout_s)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise AcpError(f"timeout waiting for response to {method}") from exc
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)
        if "error" in response:
            err = response["error"] or {}
            raise AcpError(
                f"{method}: {err.get('message', 'unknown error')} "
                f"(code={err.get('code', '?')})"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            return {}
        return result

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            if not raw:
                continue
            try:
                msg = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                rid = msg.get("id")
                if not isinstance(rid, str):
                    continue
                with self._pending_lock:
                    q = self._pending.get(rid)
                if q is not None:
                    try:
                        q.put_nowait(msg)
                    except queue.Full:
                        pass
                continue
            method = msg.get("method")
            if method == "session/update":
                params = msg.get("params")
                if isinstance(params, dict):
                    self._update_queue.put(params)
            # other server-originated requests/notifications (e.g. fs reads,
            # permission asks) are intentionally not handled in v1.
        # stdout closed: cancel any pending requests so callers don't hang
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for q in pending:
            try:
                q.put_nowait({"id": "", "error": {"code": -32000, "message": "child exited"}})
            except queue.Full:
                pass

    def _stderr_drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for _ in proc.stderr:
            # silently consume so the pipe never blocks; users can rerun under
            # `RUST_LOG=info` and direct stderr to a file if they need the log
            pass

    def _drain_updates(self) -> None:
        while True:
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                return


def _coerce_session_id(result: dict[str, Any]) -> str | None:
    sid = result.get("sessionId")
    if isinstance(sid, str):
        return sid
    sid = result.get("session_id")
    if isinstance(sid, str):
        return sid
    return None
