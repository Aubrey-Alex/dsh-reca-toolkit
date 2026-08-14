"""Configuration and local runtime checks for ReCA Director."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _dotenv_keys(path: Path) -> dict[str, bool]:
    """Read only presence of non-empty dotenv values; never expose values."""
    values: dict[str, bool] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = bool(value)
    except OSError:
        pass
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks: list[dict[str, str]] = []
    checks.append({"name": "python", "status": "ok", "detail": sys.version.split()[0]})
    checks.append({"name": "ffmpeg", "status": "ok" if shutil.which("ffmpeg") else "missing", "detail": shutil.which("ffmpeg") or "not found"})
    checks.append({"name": "dsh", "status": "ok" if shutil.which("dsh") else "missing", "detail": shutil.which("dsh") or "not found"})
    env_path = root / ".env"
    checks.append({"name": ".env", "status": "ok" if env_path.is_file() else "missing", "detail": str(env_path)})
    dotenv = _dotenv_keys(env_path)
    for key in ("RECA_PLANNER_API_KEY", "RECA_WAN30_API_KEY", "RECA_GPT_API_KEY"):
        present = bool(os.environ.get(key)) or dotenv.get(key, False)
        checks.append({"name": key, "status": "set" if present else "not_set", "detail": "value hidden"})
    print(json.dumps({"checks": checks}, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] not in {"missing"} for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
