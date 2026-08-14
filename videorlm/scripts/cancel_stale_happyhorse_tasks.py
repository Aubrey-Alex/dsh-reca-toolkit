"""Cancel stale happyhorse-1.0-r2v tasks recorded in run/logs/trace_log.txt.

Use cases:
  * After a process is killed mid-run, free queue slots so the next
    submit doesn't wait behind our own zombies.
  * After bumping ``RECA_HAPPYHORSE_POLL_MAX_WAIT_S`` for the first time —
    one-time drain of the pre-fix backlog.

A task is "stale" when it is still PENDING/RUNNING but is NOT the task_id
recorded in any ``<segment>.mp4.task_id`` sidecar — i.e. it is no longer
being polled by any in-process pipeline.

Usage:
    python3 -m videorlm.scripts.cancel_stale_happyhorse_tasks \\
        videorlm/outputs/version_2.4/ex49_happyhorse \\
        videorlm/outputs/version_2.4/ex50_happyhorse ...

    # or to cancel all PENDING/RUNNING in every cell under a parent dir
    python3 -m videorlm.scripts.cancel_stale_happyhorse_tasks \\
        --all videorlm/outputs/version_2.4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


_TASK_URL_TPL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
_CANCEL_URL_TPL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}/cancel"


def _keys() -> list[str]:
    env = (
        os.environ.get("DASHSCOPE_API_KEYS")
        or os.environ.get("DASHSCOPE_API_KEY", "")
    )
    return [k.strip() for k in env.split(",") if k.strip()]


def _live_task_ids(cell_dir: Path) -> set[str]:
    """Task ids that are currently being polled by some pipeline process."""
    out: set[str] = set()
    seg_dir = cell_dir / "run" / "segments"
    if not seg_dir.exists():
        return out
    for p in seg_dir.glob("*.mp4.task_id"):
        try:
            out.add(p.read_text(encoding="utf-8").strip())
        except Exception:  # noqa: BLE001
            continue
    return out


def _submitted_task_ids(cell_dir: Path) -> list[str]:
    """All task_ids ever submitted by this cell (from trace_log)."""
    log = cell_dir / "run" / "logs" / "trace_log.txt"
    out: list[str] = []
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("event") != "task_submitted":
            continue
        if "happyhorse" not in d.get("backend", ""):
            continue
        tid = d.get("task_id")
        if tid:
            out.append(tid)
    return out


def _status(tid: str, key: str) -> str:
    try:
        r = httpx.get(
            _TASK_URL_TPL.format(task_id=tid),
            headers={"Authorization": f"Bearer {key}"},
            timeout=10,
        )
        return (r.json().get("output") or {}).get("task_status", f"http{r.status_code}")
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def _cancel(tid: str, keys: list[str]) -> bool:
    """POST cancel on every key — task ownership is per-key, only one wins."""
    for k in keys:
        try:
            r = httpx.post(
                _CANCEL_URL_TPL.format(task_id=tid),
                headers={"Authorization": f"Bearer {k}"},
                timeout=10,
            )
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def cancel_stale_in(cell_dir: Path) -> tuple[int, int, int]:
    keys = _keys()
    if not keys:
        print("[fatal] DASHSCOPE_API_KEY[S] not set", file=sys.stderr)
        sys.exit(2)
    live = _live_task_ids(cell_dir)
    submitted = _submitted_task_ids(cell_dir)
    candidates = [t for t in submitted if t not in live]
    if not candidates:
        return 0, 0, 0

    # Query statuses in parallel
    statuses: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(_status, t, keys[0]): t for t in candidates}
        for f in as_completed(futs):
            statuses[futs[f]] = f.result()

    stale = [t for t, s in statuses.items() if s in ("PENDING", "RUNNING")]
    if not stale:
        return len(candidates), 0, 0

    # Cancel in parallel
    cancelled = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_cancel, t, keys): t for t in stale}
        for f in as_completed(futs):
            if f.result():
                cancelled += 1
    return len(candidates), len(stale), cancelled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", help="cell out_dirs to clean")
    ap.add_argument("--all", help="recurse into all subdirs of this parent")
    args = ap.parse_args()

    targets: list[Path] = [Path(c) for c in args.cells]
    if args.all:
        parent = Path(args.all)
        targets.extend(
            d for d in parent.iterdir()
            if d.is_dir() and (d / "run" / "logs" / "trace_log.txt").exists()
        )
    if not targets:
        ap.error("no cells given (use positional args or --all)")

    grand_seen = grand_stale = grand_cancelled = 0
    for cell in sorted(set(targets)):
        try:
            seen, stale, cancelled = cancel_stale_in(cell)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {cell.name}: {type(e).__name__}: {e}")
            continue
        grand_seen += seen
        grand_stale += stale
        grand_cancelled += cancelled
        if seen or stale:
            print(
                f"{cell.name}: {seen} submitted-not-live, "
                f"{stale} PENDING/RUNNING, {cancelled} cancelled"
            )
    print(
        f"\n=== TOTAL: scanned={grand_seen} stale={grand_stale} "
        f"cancelled={grand_cancelled} ==="
    )


if __name__ == "__main__":
    main()
