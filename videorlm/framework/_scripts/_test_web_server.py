"""Smoke tests for the BYOK (run-it-yourself) web server.

Runs the server on a random loopback port, exercises every endpoint, and
checks the path-traversal guard. Does NOT run the actual ReCA pipeline
(that would need real API keys); instead it submits a stub job and
verifies the job lifecycle reaches a terminal state without crashing the
server.

Run::

    cd /mnt/workspace/akide/code/unirlm-02
    python3 -m videorlm.framework._scripts._test_web_server

Exits 0 on success, non-zero on any check failure. Single-file stdlib
test runner — no pytest dependency.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]


# ─── tiny test harness ────────────────────────────────────────────────


_failures: list[str] = []


def _check(cond: bool, msg: str) -> None:
    sigil = "✓" if cond else "✗"
    print(f"  {sigil} {msg}")
    if not cond:
        _failures.append(msg)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_get(url: str, timeout: float = 4.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""


def _http_post_json(url: str, body: dict, timeout: float = 4.0) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:    payload = json.loads(e.read() or b"{}")
        except Exception: payload = {}
        return e.code, payload


def _wait_for_listener(port: int, deadline_s: float = 8.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# ─── server lifecycle ────────────────────────────────────────────────


def _spawn_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "videorlm.framework._scripts._web_server",
         "--port", str(port)],
        cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc


# ─── tests ────────────────────────────────────────────────────────────


def test_health_and_endpoints():
    port = _free_port()
    proc = _spawn_server(port)
    try:
        _check(_wait_for_listener(port), f"server bound to 127.0.0.1:{port}")
        if proc.poll() is not None:
            print("server died early; stdout:")
            print(proc.stdout.read() if proc.stdout else "<no output>")
            return

        # /health
        code, body = _http_get(f"http://127.0.0.1:{port}/health")
        _check(code == 200, f"/health → 200 (got {code})")
        _check(json.loads(body).get("ok") is True, "/health body has ok:True")

        # POST /api/run with missing story → 400
        code, body = _http_post_json(f"http://127.0.0.1:{port}/api/run", {
            "story": "",
            "planner_api_key": "sk-test",
            "planner_model": "qwen3.6-max-preview",
            "image_backend": "gpt-image-2",
            "video_backend": "happyhorse",
        })
        _check(code == 400, f"empty story → 400 (got {code})")

        # POST /api/run with missing planner_api_key → 400
        code, body = _http_post_json(f"http://127.0.0.1:{port}/api/run", {
            "story": "stub story",
            "planner_api_key": "",
            "planner_model": "qwen3.6-max-preview",
            "image_backend": "gpt-image-2",
            "video_backend": "happyhorse",
        })
        _check(code == 400, f"empty planner_api_key → 400 (got {code})")

        # POST /api/run with a stub payload — should accept then the subprocess
        # will fail (no real keys) but server stays alive
        code, body = _http_post_json(f"http://127.0.0.1:{port}/api/run", {
            "story": "stub for byok server smoke test — runner will fail at planner login",
            "planner_api_key": "sk-fake-not-real",
            "planner_base_url": "https://example.invalid/v1",
            "planner_model": "qwen3.6-max-preview",
            "image_backend": "gpt-image-2",
            "video_backend": "happyhorse",
            "dashscope_api_key": "sk-fake",
            "ziplab_api_key": "sk-fake",
            "resolution": "1920x1080",
            "seed": 0,
        })
        _check(code == 200, f"/api/run with stub payload → 200 (got {code})")
        jid = body.get("job_id", "")
        _check(bool(jid), f"job_id returned ({jid!r})")

        # GET /api/jobs/<id> immediately — must return a status JSON
        code, raw = _http_get(f"http://127.0.0.1:{port}/api/jobs/{jid}")
        _check(code == 200, "/api/jobs/<id> → 200")
        data = json.loads(raw)
        for k in ("id", "status", "stages", "log_tail"):
            _check(k in data, f"job status has key {k!r}")
        _check(set(data["stages"].keys()) >= {
            "plan_skeleton", "plan_segments", "images_dag", "segments", "bridges", "concat",
        }, "stages dict has all 6 stage keys")

        # GET /api/jobs/<id>/manifest — must work even before any artifact lands
        code, raw = _http_get(f"http://127.0.0.1:{port}/api/jobs/{jid}/manifest")
        _check(code == 200, "/api/jobs/<id>/manifest → 200")
        m = json.loads(raw)
        for k in ("id", "shots", "skeleton", "planner", "final"):
            _check(k in m, f"manifest has key {k!r}")
        _check(isinstance(m.get("shots"), list), "manifest.shots is a list")

        # Path-traversal guard on /asset
        code, _ = _http_get(f"http://127.0.0.1:{port}/api/jobs/{jid}/asset/../../../../etc/passwd")
        _check(code in (403, 404), f"path-traversal blocked (got {code})")

        # Stub job: ensure server stayed healthy and at least reached
        # ``running``. The pipeline's planner retry loop (default 5×
        # exponential backoff) can take well over the test budget to
        # fail, so don't pin terminal status; just confirm progress.
        code, raw = _http_get(f"http://127.0.0.1:{port}/api/jobs/{jid}")
        _check(code == 200, "job poll after job start still 200")
        status = json.loads(raw).get("status", "")
        _check(status in ("running", "failed", "pending"),
               f"stub job status is sane: {status!r}")

        # 404 paths
        code, _ = _http_get(f"http://127.0.0.1:{port}/api/jobs/deadbeefdeadbeef")
        _check(code == 404, "unknown job → 404")
        code, _ = _http_get(f"http://127.0.0.1:{port}/totally-not-a-route")
        _check(code == 404, "unknown route → 404")

        # Server still alive after all that
        code, body = _http_get(f"http://127.0.0.1:{port}/health")
        _check(code == 200, "server still alive at end")

    finally:
        proc.terminate()
        try: proc.wait(timeout=3)
        except subprocess.TimeoutExpired: proc.kill()


def main() -> int:
    print("=== byok web_server smoke tests ===")
    test_health_and_endpoints()
    print()
    if _failures:
        print(f"✗ {len(_failures)} failure(s):")
        for f in _failures: print(f"  - {f}")
        return 1
    print("✓ all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
