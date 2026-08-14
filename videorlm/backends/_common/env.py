"""Pipeline-local .env loader for backend providers.

Backends are commonly launched from different working directories, so env
fallbacks need to cover both the pipeline-local file and the repo-root file:

``experimental_baseline/pipeline/.env``
``.env``
"""
from __future__ import annotations

from pathlib import Path
import os


def pipeline_env_path() -> Path:
    # .../pipeline/stage2_gen/ours/backends/_common/env.py
    # parents[4] == .../experimental_baseline/pipeline
    return Path(__file__).resolve().parents[4] / ".env"


def repo_env_path() -> Path:
    # parents[6] == .../unirlm-02
    return Path(__file__).resolve().parents[6] / ".env"


def env_paths() -> tuple[Path, ...]:
    here = Path(__file__).resolve()
    paths = [
        pipeline_env_path(),
        repo_env_path(),
        here.parents[5] / ".env",  # legacy experimental_baseline/.env, if any
        Path.home() / ".env",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return tuple(out)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val:
            out[key] = val
    return out


def read_pipeline_env() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in reversed(env_paths()):
        out.update(_read_env_file(path))
    return out


def env_value(*names: str) -> str:
    values = read_pipeline_env()
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    for name in names:
        val = values.get(name, "")
        if val:
            return val
    return ""
