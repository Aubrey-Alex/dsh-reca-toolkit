"""Backend trace logging helpers."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()


def _trace_path(log_dir: str | None = None) -> str | None:
    if log_dir:
        base = os.path.abspath(log_dir)
        if os.path.basename(base) == "prompts":
            base = os.path.dirname(base)
        return os.path.join(base, "trace_log.txt")
    raw = os.environ.get("RECA_BACKEND_TRACE_LOG_PATH", "").strip()
    return raw or None


def trace_event(component: str, event: str, *, log_dir: str | None = None, **fields: Any) -> None:
    path = _trace_path(log_dir)
    if not path:
        return
    payload = {
        "ts": time.time(),
        "component": component,
        "event": event,
        **fields,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return
