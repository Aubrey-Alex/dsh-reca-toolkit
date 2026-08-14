"""Local web server for the `demo/real_demo` "Run-It-Yourself" section.

A thin job-runner around ``videorlm.framework._scripts._smoke`` that the
in-browser form posts to. Pure stdlib (no FastAPI / aiohttp deps), so a
fresh clone with the framework installed can launch it directly:

    python3 -m videorlm.framework._scripts._web_server --port 18800

The browser tab at ``demo/real_demo/index.html#run-yourself`` then talks
to ``http://localhost:<port>``. Nothing is exposed beyond loopback.

Endpoints
---------
  GET  /health                  → {ok: True}
  POST /api/run                 → {job_id}
  GET  /api/jobs/<id>           → {status, stages, log_tail, final_url?}
  GET  /api/jobs/<id>/manifest  → {shots, anchors, portraits, locations, props,
                                    segments, bridges, final} — URLs of every
                                    artifact currently on disk. Frontend polls
                                    to progressively render the run like the
                                    demo showcase page.
  GET  /api/jobs/<id>/final.mp4 → binary mp4 (when status==done)
  GET  /api/jobs/<id>/asset/<path>  → any file under the job's run dir,
                                       used for inline previews (PNG/MP4).

Job runtime
-----------
Each POST /api/run spawns a Python subprocess invoking ``_smoke.py`` with
the user-supplied env vars, writing into ``./byok_jobs/<id>/``. Status is
inferred from log markers (``[stage] X START`` / ``dt=`` / ``run_render OK``).
The server is single-process; concurrent jobs are allowed (the underlying
key pool / xsem already handle backend concurrency).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[3]
JOBS_ROOT = REPO / "byok_jobs"
JOBS_ROOT.mkdir(exist_ok=True)


# ─── Job state ─────────────────────────────────────────────────────────

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

_STAGES = (
    "plan_skeleton", "plan_segments", "images_dag",
    "anchor_validator",   # only populated when --validate is on
    "segments",
    "segment_validator",  # only populated when --validate-segments is on
    "bridges", "concat",
)


def _new_job(resume_jid: str | None = None) -> str:
    """Create a fresh job, or adopt an existing job dir for --resume.

    When ``resume_jid`` is given and ``byok_jobs/<resume_jid>/`` exists, the
    new job reuses that directory (so cached PNGs / segment mp4s survive)
    and the runner appends ``--resume`` to the smoke.py command.
    """
    if resume_jid:
        candidate = JOBS_ROOT / resume_jid
        if candidate.exists() and (candidate / "story.txt").exists():
            jid = resume_jid
        else:
            jid = uuid.uuid4().hex[:12]
    else:
        jid = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    with _JOBS_LOCK:
        _JOBS[jid] = {
            "id": jid,
            "status": "pending",
            "stages": {s: "pending" for s in _STAGES},
            "log_path": str(job_dir / "run.log"),
            "out_dir": str(job_dir),
            "started_at": time.time(),
            "final_url": None,
            "resumed": bool(resume_jid and resume_jid == jid),
        }
    return jid


def _get_job(jid: str) -> dict | None:
    with _JOBS_LOCK:
        return dict(_JOBS.get(jid) or {}) if jid in _JOBS else None


def _set_status(jid: str, status: str) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS: _JOBS[jid]["status"] = status


def _set_stage(jid: str, stage: str, status: str) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS and stage in _JOBS[jid]["stages"]:
            _JOBS[jid]["stages"][stage] = status


def _set_final(jid: str, mp4_path: str) -> None:
    with _JOBS_LOCK:
        if jid in _JOBS:
            _JOBS[jid]["final_path"] = mp4_path
            _JOBS[jid]["final_url"] = f"/api/jobs/{jid}/final.mp4"


# ─── Pipeline runner (subprocess + log scraper) ───────────────────────


def _backend_env_name(backend: str) -> str:
    """Map backend display name → RECA_BACKEND_CONCURRENCY_* env suffix.

    e.g. ``qwen-image-2.0-pro`` → ``QWEN_IMAGE_2_0_PRO``;
         ``gpt-image-2``       → ``GPT_IMAGE_2``;
         ``wan2.7-image``      → ``WAN2_7_IMAGE``
    """
    return backend.upper().replace("-", "_").replace(".", "_")


def _build_env(form: dict) -> dict[str, str]:
    """Map form payload → env vars for _smoke.py."""
    env = os.environ.copy()
    api_path = "messages" if form.get("planner_model", "").startswith("gpt-5") else "chat"

    env.update({
        "RECA_DISABLE_HTTP_PROXY": "1",
        "RECA_XSEM_ENABLE": "1",
        "RECA_HAPPYHORSE_POLL_MAX_WAIT_S": "3600",
        "RECA_VIDEO_CALL_TIMEOUT_S": "3700",
        "RECA_XSEM_HAPPYHORSE_R2V": "8",
        "RECA_PLAN_SKELETON_MAX_RETRIES": "8",
        "RECA_PLANNER_MODEL": form["planner_model"],
        "RECA_PLANNER_API_PATH": api_path,
        "RECA_PLANNER_API_KEY": form.get("planner_api_key", ""),
        "RECA_PLANNER_BASE_URL": form.get("planner_base_url", ""),
        "RECA_RENDER_BACKEND_PORTRAIT":     form["image_backend"],
        "RECA_RENDER_BACKEND_ANCHOR_IMAGE": form["image_backend"],
        "RECA_RENDER_BACKEND_IMAGE_EDIT":   form["image_backend"],
        "DASHSCOPE_API_KEY":  form.get("dashscope_api_key", ""),
        "DASHSCOPE_API_KEYS": form.get("dashscope_api_key", ""),
        "OPENAI_API_KEY":     form.get("ziplab_api_key", ""),
        "OPENAI_BASE_URL":    form.get("planner_base_url", ""),
        "ZIPLAB_API_KEY":     form.get("ziplab_api_key", ""),
    })
    if api_path == "messages":
        env["RECA_PLANNER_THINKING_BUDGET"] = "4000"
    # Image-backend concurrency knob (RECA_BACKEND_CONCURRENCY_<NAME>)
    n = int(form.get("image_concurrency") or 0)
    if n > 0:
        env[f"RECA_BACKEND_CONCURRENCY_{_backend_env_name(form['image_backend'])}"] = str(n)
    # Image-backend fallback chain for content-moderation rejection
    fallback = (form.get("image_fallback_chain") or "").strip()
    if fallback:
        env["RECA_IMAGE_FALLBACK_CHAIN"] = fallback
    return env


_STAGE_PATTERNS = [
    (re.compile(r"\[stage\] (images-dag) +START"),         ("images_dag", "running")),
    (re.compile(r"\[stage\] (images-dag) +dt="),           ("images_dag", "done")),
    (re.compile(r"\[stage\] (anchor-validator) +START"),   ("anchor_validator", "running")),
    (re.compile(r"\[stage\] (anchor-validator) +dt="),     ("anchor_validator", "done")),
    (re.compile(r"\[stage\] (segments) +START"),           ("segments", "running")),
    (re.compile(r"\[stage\] (segments) +dt="),             ("segments", "done")),
    (re.compile(r"\[stage\] (segments) +dt="),             ("segment_validator", "done")),
    # Segment validator runs in-line inside _render_shot_chain (no standalone
    # stage line), so flip to running on the first [segment-validate] /
    # [router] / [seg-judgment] log line; done when segments stage closes.
    (re.compile(r"\[(segment-validate|router|seg-judgment|seg-validator)\]"),
                                                           ("segment_validator", "running")),
    (re.compile(r"\[stage\] (bridges) +START"),            ("bridges", "running")),
    (re.compile(r"\[stage\] (bridges) +dt="),              ("bridges", "done")),
    (re.compile(r"\[stage\] (concat) +START"),             ("concat", "running")),
    (re.compile(r"\[stage\] (concat) +dt="),               ("concat", "done")),
    (re.compile(r"plan_skeleton OK"),                      ("plan_skeleton", "done")),
    (re.compile(r"plan_segments_all OK"),                  ("plan_segments", "done")),
    (re.compile(r"plan_skeleton.*attempt 1"),              ("plan_skeleton", "running")),
    (re.compile(r"plan_segments_all OK; segments="),       ("plan_segments", "done")),
    (re.compile(r"run_render OK"),                         ("concat", "done")),
]


def _scrape_stages(jid: str, line: str) -> None:
    for pat, (stage, status) in _STAGE_PATTERNS:
        if pat.search(line):
            _set_stage(jid, stage, status)


def _run_job(jid: str, form: dict) -> None:
    job = _get_job(jid)
    if not job: return
    out_dir = Path(job["out_dir"])
    log_path = Path(job["log_path"])
    story_path = out_dir / "story.txt"
    story_path.write_text(form.get("story", ""), encoding="utf-8")

    env = _build_env(form)
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        sys.executable, "-u", "-m", "videorlm.framework._scripts._smoke",
        "--story", str(story_path),
        "--out-dir", str(out_dir),
        "--label", f"byok-{jid}",
        "--segments", "--render",
        "--backend", form["video_backend"],
        "--video-resolution", form.get("resolution", "1920x1080"),
        "--seed", str(int(form.get("seed", 0) or 0)),
    ]
    if form.get("validate"):
        cmd.append("--validate")
    if form.get("validate_segments"):
        cmd.append("--validate-segments")
    if (job.get("resumed") or form.get("resume_job_id")) and (out_dir / "render_plan.json").exists():
        cmd.append("--resume")
    _set_status(jid, "running")
    _set_stage(jid, "plan_skeleton", "running")

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# job {jid}\n# cmd: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            _scrape_stages(jid, line)
        proc.wait()

    rc = proc.returncode
    final = out_dir / "run" / "final.mp4"
    if rc == 0 and final.exists():
        _set_final(jid, str(final))
        # Final concat marker comes from "run_render OK" but only when it happens.
        # We also force-flip remaining stages to done in case scraping missed them.
        with _JOBS_LOCK:
            for s in _STAGES:
                if _JOBS[jid]["stages"][s] != "done":
                    _JOBS[jid]["stages"][s] = "done"
        _set_status(jid, "done")
    else:
        with _JOBS_LOCK:
            for s in _STAGES:
                if _JOBS[jid]["stages"][s] == "running":
                    _JOBS[jid]["stages"][s] = "failed"
        _set_status(jid, "failed")


# ─── Manifest scanner ─────────────────────────────────────────────────


_KIND_DIRS = (
    ("portraits", "portraits"),
    ("locations", "locations"),
    ("props",     "props"),
    ("anchors",   "anchors"),
    ("segments",  "segments"),
    ("bridges",   "bridges"),
)


def _scan_manifest(jid: str, out_dir: Path) -> dict:
    """Walk job's run dir; return URLs (browser-fetchable) for every artifact.

    Canonical layout produced by _smoke.py (--out-dir=<JOB>):
      <JOB>/{skeleton,planner,render_plan}.json   ← top-level planning JSONs
      <JOB>/run/{portraits,locations,props,anchors,segments,bridges}/*
      <JOB>/run/final.mp4                         ← final concat
    """
    run = out_dir / "run"
    manifest: dict = {
        "id": jid,
        "skeleton": None,
        "planner":  None,
        "render_plan": None,
        "final":    None,
        "shots":    [],
    }
    for key, name in (("skeleton", "skeleton.json"),
                      ("planner",  "planner.json"),
                      ("render_plan", "render_plan.json")):
        p = out_dir / name
        if p.exists():
            manifest[key] = f"/api/jobs/{jid}/asset/{name}"
            if key in ("skeleton", "planner"):
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                    shots = j.get("shots") or []
                    if shots and not manifest["shots"]:
                        manifest["shots"] = [
                            {"id": s.get("id", f"shot{i:02d}"),
                             "duration_s": s.get("duration_s", 0),
                             "story_goal": (s.get("story_goal") or "")[:160]}
                            for i, s in enumerate(shots)
                        ]
                except Exception: pass
    final = run / "final.mp4"
    if final.exists():
        manifest["final"] = f"/api/jobs/{jid}/asset/run/final.mp4"
    for key, sub in _KIND_DIRS:
        d = run / sub
        if not d.exists(): continue
        items: list[dict] = []
        for f in sorted(d.iterdir()):
            if f.name.endswith((".png", ".mp4")) and not f.name.endswith(".tail.png"):
                items.append({
                    "id":  f.stem,
                    "name": f.name,
                    "url": f"/api/jobs/{jid}/asset/run/{sub}/{f.name}",
                })
        manifest[key] = items
    return manifest


_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


def _guess_content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


# ─── HTTP handler ─────────────────────────────────────────────────────


_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
}


def _read_log_tail(path: str, n_bytes: int = 16384) -> str:
    p = Path(path)
    if not p.exists(): return ""
    sz = p.stat().st_size
    with p.open("rb") as f:
        f.seek(max(0, sz - n_bytes))
        return f.read().decode("utf-8", errors="replace")


class _Handler(BaseHTTPRequestHandler):
    server_version = "RecaByok/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[byok-server] {self.client_address[0]} {fmt % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        for k, v in _CORS_HEADERS.items(): self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        for k, v in _CORS_HEADERS.items(): self.send_header(k, v)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for k, v in _CORS_HEADERS.items(): self.send_header(k, v)
        self.end_headers()

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/health":
            return self._send_json(HTTPStatus.OK, {"ok": True})

        # /api/jobs/<id>/manifest — progressive asset list (PNG/MP4/JSON)
        m = re.match(r"^/api/jobs/([a-f0-9]+)/manifest$", url.path)
        if m:
            jid = m.group(1)
            job = _get_job(jid)
            if not job: return self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such job"})
            return self._send_json(HTTPStatus.OK, _scan_manifest(jid, Path(job["out_dir"])))

        # /api/jobs/<id>/asset/<relpath> — inline preview of any artifact
        m = re.match(r"^/api/jobs/([a-f0-9]+)/asset/(.+)$", url.path)
        if m:
            jid, rel = m.group(1), m.group(2)
            job = _get_job(jid)
            if not job: return self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such job"})
            base = Path(job["out_dir"]).resolve()
            target = (base / rel).resolve()
            # path-traversal guard
            try: target.relative_to(base)
            except ValueError: return self._send_json(HTTPStatus.FORBIDDEN, {"error": "bad path"})
            if not target.exists() or not target.is_file():
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "asset missing"})
            ct = _guess_content_type(target)
            return self._send_bytes(HTTPStatus.OK, target.read_bytes(), ct)

        # /api/jobs/<id> | /api/jobs/<id>/final.mp4
        m = re.match(r"^/api/jobs/([a-f0-9]+)(/final\.mp4)?$", url.path)
        if m:
            jid = m.group(1)
            job = _get_job(jid)
            if not job: return self._send_json(HTTPStatus.NOT_FOUND, {"error": "no such job"})
            if m.group(2):
                # final.mp4 download
                final = job.get("final_path")
                if not final or not Path(final).exists():
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "final not ready"})
                data = Path(final).read_bytes()
                return self._send_bytes(HTTPStatus.OK, data, "video/mp4")
            # status JSON
            return self._send_json(HTTPStatus.OK, {
                "id": jid,
                "status": job["status"],
                "stages": job["stages"],
                "log_tail": _read_log_tail(job["log_path"]),
                "final_url": job.get("final_url"),
            })

        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/run":
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 256 * 1024:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad payload size"})
        raw = self.rfile.read(length)
        try:
            form = json.loads(raw)
        except Exception as e:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid json: {e}"})
        if not form.get("story", "").strip():
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "story is empty"})
        if not form.get("planner_api_key", "").strip():
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "planner_api_key required"})

        jid = _new_job(resume_jid=form.get("resume_job_id") or None)
        t = threading.Thread(target=_run_job, args=(jid, form), daemon=True)
        t.start()
        return self._send_json(HTTPStatus.OK, {"job_id": jid})


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18800, help="listen port (default 18800)")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1, loopback only)")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[byok-server] listening on http://{args.host}:{args.port}", flush=True)
    print(f"[byok-server] open demo/real_demo/index.html#run-yourself in your browser", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[byok-server] shutting down", flush=True)


if __name__ == "__main__":
    main()
