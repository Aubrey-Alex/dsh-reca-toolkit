"""End-to-end simulation of the demo page's interactive run flow.

Acts as the browser would: GET /health, POST /api/run with a real payload,
poll /api/jobs/<id> + /api/jobs/<id>/manifest every few seconds, log every
stage transition and every new asset that appears. Verifies that the page
WOULD render shots → portraits → anchors → segments → final.mp4 in real
time.

Defaults to a "fast" config (qwen3.6 planner + qwen-image-2.0-pro anchors
+ happyhorse video) to reach final.mp4 within roughly 30-60 min on a real
backend. Keys come from .env.

Run::

    cd /mnt/workspace/akide/code/unirlm-02
    python3 -m videorlm.framework._scripts._test_demo_e2e \\
        --story input_examples/example_01.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = "http://localhost:18800"

REPO = Path(__file__).resolve().parents[3]


# ─── HTTP helpers ────────────────────────────────────────────────────


def _get(url: str, timeout: float = 10.0, *, retries: int = 4) -> dict:
    """Transient-aware GET. Treats 5xx + connection errors as retryable
    (server briefly busy when downstream subprocess saturates IO)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if 500 <= e.code < 600 and attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"unreachable: _get({url}) exhausted {retries} retries")


def _post(url: str, body: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _load_dotenv() -> dict[str, str]:
    out: dict[str, str] = {}
    env = REPO / ".env"
    if not env.exists(): return out
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ─── E2E run ─────────────────────────────────────────────────────────


def _now() -> str: return time.strftime("%H:%M:%S")


def _summarize_manifest(m: dict) -> dict[str, int]:
    return {
        "shots":     len(m.get("shots") or []),
        "portraits": len(m.get("portraits") or []),
        "locations": len(m.get("locations") or []),
        "props":     len(m.get("props") or []),
        "anchors":   len(m.get("anchors") or []),
        "segments":  len(m.get("segments") or []),
        "bridges":   len(m.get("bridges") or []),
        "final":     1 if m.get("final") else 0,
    }


def run_e2e(args: argparse.Namespace) -> int:
    env = _load_dotenv()
    story = Path(args.story).read_text(encoding="utf-8").strip()
    print(f"[{_now()}] simulating demo flow against {SERVER}")
    print(f"[{_now()}] story: {args.story} ({len(story)} chars)")

    # Step 1: health
    try:
        h = _get(f"{SERVER}/health")
        print(f"[{_now()}] ✓ GET /health → {h}")
    except urllib.error.URLError as e:
        print(f"[{_now()}] ✗ server unreachable: {e}")
        print("                start it:  python3 -m videorlm.framework._scripts._web_server --port 18800")
        return 2

    # Step 2: submit (POST /api/run)
    payload = {
        "story": story,
        "planner_model": args.planner_model,
        "planner_base_url": args.planner_base_url,
        "planner_api_key": env.get("DASHSCOPE_API_KEY", ""),
        "image_backend": args.image_backend,
        "video_backend": args.video_backend,
        "dashscope_api_key": env.get("DASHSCOPE_API_KEY", ""),
        "ziplab_api_key": env.get("OPENAI_API_KEY", ""),
        "resolution": args.resolution,
        "seed": args.seed,
    }
    if not payload["planner_api_key"]:
        print(f"[{_now()}] ✗ DASHSCOPE_API_KEY missing in .env")
        return 2
    print(f"[{_now()}] POSTing /api/run  "
          f"planner={payload['planner_model']} image={payload['image_backend']} video={payload['video_backend']}")
    resp = _post(f"{SERVER}/api/run", payload, timeout=15)
    jid = resp.get("job_id")
    if not jid:
        print(f"[{_now()}] ✗ no job_id in response: {resp}")
        return 1
    print(f"[{_now()}] ✓ job_id = {jid}")

    # Step 3 + 4: poll status + manifest, log every transition
    prev_stages: dict[str, str] = {}
    prev_counts: dict[str, int] = {}
    deadline = time.time() + args.deadline_min * 60
    while time.time() < deadline:
        # status
        try:
            s = _get(f"{SERVER}/api/jobs/{jid}", timeout=8)
        except Exception as e:
            print(f"[{_now()}] ✗ /api/jobs/{jid} → {type(e).__name__}: {e}")
            return 1
        stages = s.get("stages", {})
        for k, v in stages.items():
            if prev_stages.get(k) != v:
                print(f"[{_now()}]   stage {k:14s}: {prev_stages.get(k,'?'):>8s} → {v}")
                prev_stages[k] = v

        # manifest
        try:
            m = _get(f"{SERVER}/api/jobs/{jid}/manifest", timeout=8)
        except Exception:
            m = {}
        counts = _summarize_manifest(m)
        deltas = {k: counts[k] - prev_counts.get(k, 0) for k in counts if counts[k] != prev_counts.get(k, 0)}
        if deltas:
            tags = " ".join(f"{k}={counts[k]}(+{deltas[k]})" for k in deltas)
            print(f"[{_now()}]   assets: {tags}")
            prev_counts = dict(counts)

        if s.get("status") == "done":
            print(f"[{_now()}] ✓ DONE — final_url={s.get('final_url')}")
            print(f"[{_now()}]   final counts: shots={counts['shots']} anchors={counts['anchors']} "
                  f"segments={counts['segments']} bridges={counts['bridges']}")
            return 0
        if s.get("status") == "failed":
            print(f"[{_now()}] ✗ FAILED")
            print(f"[{_now()}]   last log tail:")
            for line in (s.get("log_tail") or "").splitlines()[-20:]:
                print(f"      | {line}")
            return 1

        time.sleep(args.poll_s)

    print(f"[{_now()}] ✗ deadline {args.deadline_min}min reached without terminal status")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default=str(REPO / "input_examples" / "example_01.txt"))
    ap.add_argument("--planner-model", default="qwen3.6-max-preview")
    ap.add_argument("--planner-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--image-backend", default="qwen-image-2.0-pro",
                    choices=["gpt-image-2", "qwen-image-2.0-pro", "wan2.7-image"])
    ap.add_argument("--video-backend", default="happyhorse",
                    choices=["happyhorse", "wan", "ltx"])
    ap.add_argument("--resolution", default="1920x1080")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll-s", type=float, default=4.0)
    ap.add_argument("--deadline-min", type=int, default=120)
    return run_e2e(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
