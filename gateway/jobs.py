from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .artifacts import public_manifest
from .recovery import recover_unfinished_runs
from .schemas import normalize_run_config


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("RECA_RUNS_ROOT", str(ROOT / ".dsh_runs")))
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

STAGES = (
    "plan_skeleton",
    "plan_segments",
    "images_dag",
    "anchor_validator",
    "segments",
    "segment_validator",
    "bridges",
    "concat",
)

SAFE_OPTION_KEYS = {
    "backend",
    "resolution",
    "seed",
    "validate",
    "validate_segments",
    "force_i2v",
    "max_repair_attempts",
    "resume_run_id",
    "duration",
    "duration_s",
    "style",
    "aspect_ratio",
    "enable_audit",
    "run_id",
    "label",
}

RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$", re.IGNORECASE)

_STAGE_PATTERNS = (
    (re.compile(r"plan_skeleton.*attempt 1"), "plan_skeleton", "running"),
    (re.compile(r"plan_skeleton OK"), "plan_skeleton", "done"),
    (re.compile(r"plan_segments_all OK"), "plan_segments", "done"),
    (re.compile(r"\[stage\] images-dag +START"), "images_dag", "running"),
    (re.compile(r"\[stage\] images-dag +dt="), "images_dag", "done"),
    (re.compile(r"\[stage\] anchor-validator +START"), "anchor_validator", "running"),
    (re.compile(r"\[stage\] anchor-validator +dt="), "anchor_validator", "done"),
    (re.compile(r"\[stage\] segments +START"), "segments", "running"),
    (re.compile(r"\[stage\] segments +dt="), "segments", "done"),
    (re.compile(r"\[(segment-validate|router|seg-judgment|seg-validator)\]"),
     "segment_validator", "running"),
    (re.compile(r"\[stage\] bridges +START"), "bridges", "running"),
    (re.compile(r"\[stage\] bridges +dt="), "bridges", "done"),
    (re.compile(r"\[stage\] concat +START"), "concat", "running"),
    (re.compile(r"\[stage\] concat +dt="), "concat", "done"),
    (re.compile(r"run_render OK"), "concat", "done"),
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,]+"),
    re.compile(r"(?i)([?&](?:signature|ossaccesskeyid|accesskeyid)=)[^&\s]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def slugify_run_id(raw: str) -> str:
    text = (raw or "").strip().lower().replace("·", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"[_-]{2,}", "_", text).strip("_-")
    if not text:
        return ""
    if text[0].isdigit():
        text = f"run_{text}"
    return text[:48]


def story_run_slug(story: str) -> str:
    first = (story or "").strip().splitlines()[0] if (story or "").strip() else ""
    return slugify_run_id(first) or "film"


def allocate_run_id(runs_root: Path, base: str) -> str:
    base = slugify_run_id(base) or "film"
    candidate = base
    serial = 2
    while (runs_root / candidate).exists():
        suffix = f"_{serial}"
        candidate = f"{base[: 63 - len(suffix)]}{suffix}"
        serial += 1
        if serial > 99:
            raise RuntimeError(f"too many run folders named {base}")
    return candidate


def _redact_log(value: str) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda match: f"{match.group(1)}<redacted>", result)
        else:
            result = pattern.sub("<redacted>", result)
    return result


def _now() -> float:
    return time.time()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class JobManager:
    """Run one unchanged ReCA smoke pipeline per isolated child process."""

    def __init__(self, root: Path = ROOT, runs_root: Path = RUNS_ROOT) -> None:
        self.root = root
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        recovered = recover_unfinished_runs(self.runs_root)
        if recovered:
            print(f"[reca-gateway] recovered interrupted runs: {', '.join(recovered)}", flush=True)

    def _job_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    def _state_path(self, run_id: str) -> Path:
        return self._job_dir(run_id) / "state.json"

    def _read_state(self, run_id: str) -> dict[str, Any] | None:
        path = self._state_path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_state(self, run_id: str, state: dict[str, Any]) -> None:
        _atomic_json(self._state_path(run_id), state)

    def _update_state(self, run_id: str, **changes: Any) -> dict[str, Any] | None:
        with self._lock:
            state = self._read_state(run_id)
            if state is None:
                return None
            state.update(changes)
            self._write_state(run_id, state)
            return state

    def start(self, request: dict[str, Any]) -> dict[str, Any]:
        story = str(request.get("story") or request.get("narrative") or "").strip()
        if not story:
            raise ValueError("story is required")

        raw_options = request.get("options")
        if not isinstance(raw_options, dict):
            raw_options = {}
        raw_resume_id = raw_options.get("resume_run_id")
        resume_run_id = str(raw_resume_id).strip() if raw_resume_id else ""
        if resume_run_id:
            if not RUN_ID_RE.fullmatch(resume_run_id):
                raise ValueError("resume_run_id is invalid")
            previous = self._read_state(resume_run_id)
            if previous is None:
                raise ValueError("resume_run_id was not found")
            if previous.get("state") in {"queued", "running", "cancelling"}:
                raise ValueError("the requested run is still active")
            run_id = resume_run_id
            job_dir = self._job_dir(run_id)
            job_dir.mkdir(parents=True, exist_ok=True)
        else:
            requested = slugify_run_id(
                str(raw_options.get("run_id") or raw_options.get("label") or "")
            ) or story_run_slug(story)
            run_id = allocate_run_id(self.runs_root, requested)
            job_dir = self._job_dir(run_id)
            job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / "story.txt").write_text(story, encoding="utf-8")

        config = normalize_run_config(raw_options)
        options = {key: raw_options[key] for key in SAFE_OPTION_KEYS if key in raw_options}
        options.update(config.to_dict())
        options["validate"] = config.enable_audit
        options["validate_segments"] = config.validate_segments
        if resume_run_id:
            options["resume_run_id"] = resume_run_id
        # Keep provider credentials out of the HTTP protocol. They are loaded
        # from the process environment / ignored .env file by ReCA itself.
        safe_request = {"story": story, "options": options}
        (job_dir / "request.json").write_text(
            json.dumps(safe_request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (job_dir / "run_config.json").write_text(
            json.dumps({"run_id": run_id, **config.to_dict()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        state: dict[str, Any] = {
            "run_id": run_id,
            "state": "queued",
            "stage": "queued",
            "stages": {stage: "pending" for stage in STAGES},
            "progress": 0.0,
            "story_chars": len(story),
            "created_at": _now(),
            "started_at": None,
            "ended_at": None,
            "output_dir": str(job_dir),
            "log_file": str(job_dir / "run.log"),
            "events_file": str(job_dir / "events.jsonl"),
            "final_video": None,
            "error": None,
            "options": options,
            "gateway_state": "queued",
            "reca_state": None,
            "audit_state": "audit_pending" if config.enable_audit else "audit_skipped",
            "video_state": "pending",
        }
        self._write_state(run_id, state)
        thread = threading.Thread(
            target=self._run, args=(run_id, options), name=f"reca-{run_id}", daemon=True
        )
        thread.start()
        return self.status(run_id) or state

    def _build_command(self, run_id: str, options: dict[str, Any]) -> list[str]:
        job_dir = self._job_dir(run_id)
        backend = str(options.get("backend") or os.environ.get("RECA_DEMO_BACKEND", "wan"))
        resolution = str(options.get("resolution") or os.environ.get(
            "RECA_DEMO_RESOLUTION", "1280x720"
        ))
        seed = int(options.get("seed", 0) or 0)
        command = [
            sys.executable,
            "-u",
            "-m",
            "videorlm.framework._scripts._smoke",
            "--story",
            str(job_dir / "story.txt"),
            "--out-dir",
            str(job_dir),
            "--label",
            f"dsh-{run_id}",
            "--segments",
            "--render",
            "--backend",
            backend,
            "--video-resolution",
            resolution,
            "--seed",
            str(seed),
            "--director-config",
            str(job_dir / "run_config.json"),
        ]
        if bool(options.get("enable_audit", options.get("validate", True))):
            command.append("--validate")
        if bool(options.get("validate_segments", False)):
            command.append("--validate-segments")
        if bool(options.get("force_i2v", False)):
            command.append("--force-i2v")
        if options.get("max_repair_attempts") is not None:
            command.extend(["--max-repair-attempts", str(int(options["max_repair_attempts"]))])
        if options.get("resume_run_id") and (job_dir / "render_plan.json").exists():
            command.append("--resume")
        return command

    def _run(self, run_id: str, options: dict[str, Any]) -> None:
        job_dir = self._job_dir(run_id)
        log_path = job_dir / "run.log"
        events_path = job_dir / "events.jsonl"
        queued = self._read_state(run_id)
        if queued and queued.get("state") == "cancelling":
            self._update_state(run_id, state="cancelled", stage="cancelled", ended_at=_now())
            return
        command = self._build_command(run_id, options)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        self._update_state(
            run_id,
            state="running",
            gateway_state="running",
            stage="plan_skeleton",
            started_at=_now(),
            command=command,
        )
        self._update_stage(run_id, "plan_skeleton", "running")

        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
            with self._lock:
                self._processes[run_id] = process
            with log_path.open("w", encoding="utf-8") as log, events_path.open(
                "w", encoding="utf-8"
            ) as events:
                log.write("# command: " + " ".join(command) + "\n")
                assert process.stdout is not None
                for line in process.stdout:
                    safe_line = _redact_log(line)
                    log.write(safe_line)
                    log.flush()
                    event = {"ts": _now(), "type": "log", "text": safe_line.rstrip("\n")}
                    events.write(json.dumps(event, ensure_ascii=False) + "\n")
                    events.flush()
                    self._consume_line(run_id, safe_line)
                process.wait()
            return_code = process.returncode
        except Exception as exc:
            return_code = -1
            self._update_state(run_id, error=f"{type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._processes.pop(run_id, None)

        state = self._read_state(run_id) or {}
        cancelled = state.get("state") == "cancelling"
        final = job_dir / "run" / "final.mp4"
        if cancelled:
            terminal = "cancelled"
            error = state.get("error")
        elif return_code == 0 and final.exists():
            terminal = "succeeded"
            error = None
        else:
            terminal = "failed"
            error = state.get("error") or f"ReCA exited with code {return_code}"
        stages = state.get("stages") or {}
        if terminal == "succeeded":
            stages = {name: "done" for name in STAGES}
        else:
            stages = {
                name: ("failed" if value == "running" else value)
                for name, value in stages.items()
            }
        self._update_state(
            run_id,
            state=terminal,
            gateway_state=terminal,
            stage="done" if terminal == "succeeded" else terminal,
            stages=stages,
            progress=1.0 if terminal == "succeeded" else state.get("progress", 0.0),
            ended_at=_now(),
            final_video=(self.artifact_url(run_id, "run/final.mp4") if final.exists() else None),
            error=error,
        )

    def _consume_line(self, run_id: str, line: str) -> None:
        self._sync_reca_state(run_id)

    def _sync_reca_state(self, run_id: str) -> None:
        state = self._read_state(run_id)
        if state is None:
            return
        path = self._job_dir(run_id) / "run" / "reca_state.json"
        try:
            reca = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(reca, dict):
            return
        changes: dict[str, Any] = {
            "reca_state": reca,
            "reca_stage": reca.get("stage"),
            "audit_state": reca.get("audit_state"),
            "video_state": reca.get("video_state"),
        }
        if reca.get("progress") is not None:
            changes["reca_progress"] = reca["progress"]
        self._update_state(run_id, **changes)

    def _update_stage(self, run_id: str, stage: str, status: str) -> None:
        with self._lock:
            state = self._read_state(run_id)
            if state is None:
                return
            stages = state.get("stages") or {name: "pending" for name in STAGES}
            stages[stage] = status
            completed = sum(value == "done" for value in stages.values())
            state.update({
                "stage": stage,
                "stages": stages,
                "progress": round(completed / len(STAGES), 3),
            })
            self._write_state(run_id, state)

    def status(self, run_id: str) -> dict[str, Any] | None:
        self._sync_reca_state(run_id)
        state = self._read_state(run_id)
        if state is None:
            return None
        log_path = Path(state.get("log_file", ""))
        public = dict(state)
        for private_key in ("output_dir", "log_file", "events_file", "command"):
            public.pop(private_key, None)
        public["events_url"] = f"/v1/runs/{run_id}/events"
        if log_path.exists():
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 12000))
                public["log_tail"] = handle.read().decode("utf-8", errors="replace")
        public["artifact_manifest"] = public_manifest(
            self._job_dir(run_id), run_id, self.public_base_url()
        )
        return public

    def public_base_url(self) -> str:
        return os.environ.get(
            "RECA_PUBLIC_BASE_URL",
            f"http://{os.environ.get('RECA_GATEWAY_HOST', '127.0.0.1')}:{os.environ.get('RECA_GATEWAY_PORT', '8787')}",
        ).rstrip("/")

    def list_runs(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for state_path in sorted(self.runs_root.glob("*/state.json"), reverse=True):
            run_id = state_path.parent.name
            item = self.status(run_id)
            if item is not None:
                result.append(item)
        return result

    def resume(self, run_id: str) -> dict[str, Any] | None:
        state = self._read_state(run_id)
        if state is None:
            return None
        if state.get("state") not in {"failed", "cancelled", "interrupted"}:
            raise ValueError("only failed, cancelled, or interrupted runs can be resumed")
        request_path = self._job_dir(run_id) / "request.json"
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("run request is missing or invalid") from exc
        request.setdefault("options", {})["resume_run_id"] = run_id
        return self.start(request)

    def events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]] | None:
        state = self._read_state(run_id)
        if state is None:
            return None
        path = Path(state.get("events_file", ""))
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        result: list[dict[str, Any]] = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        state = self._read_state(run_id)
        if state is None:
            return None
        if state.get("state") in {"succeeded", "failed", "cancelled"}:
            return self.status(run_id)
        self._update_state(run_id, state="cancelling", error="cancel requested")
        with self._lock:
            process = self._processes.get(run_id)
        if process is not None:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            threading.Thread(
                target=self._kill_after_timeout,
                args=(run_id, process),
                daemon=True,
                name=f"reca-kill-{run_id}",
            ).start()
        return self.status(run_id)

    def _kill_after_timeout(self, run_id: str, process: subprocess.Popen[str]) -> None:
        timeout = float(os.environ.get("RECA_CANCEL_GRACE_S", "15"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.25)
        if process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                return

    def artifact_path(self, run_id: str, relative: str) -> Path | None:
        state = self._read_state(run_id)
        if state is None:
            return None
        base = Path(state["output_dir"]).resolve()
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return None
        return target if target.is_file() else None

    def artifact_url(self, run_id: str, relative: str) -> str:
        return f"{self.public_base_url()}/v1/runs/{run_id}/artifacts/{relative.lstrip('/') }"
