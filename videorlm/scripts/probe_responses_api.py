"""Probe OpenAI /v1/responses API via the OpenAI-compatible gateway for plan_skeleton.

Direct SSE-level instrumentation:
  - Time-stamp every event arrival.
  - Categorize event type.
  - Accumulate output_text deltas into final text.
  - Print inter-arrival gaps so we can see if reasoning_summary.delta
    keeps the connection alive (vs the chat.completions translation
    which loses reasoning events and triggers gateway's 180s timeout).

Usage:
    python3 -m videorlm.scripts.probe_responses_api
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
STORY = REPO / "input_examples" / "true_input" / "example_01.txt"


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    url = base_url + "/responses"
    if not api_key:
        raise SystemExit("missing key (set $OPENAI_API_KEY)")

    from videorlm.framework.shot_planner.prompts import PARENT_SYSTEM_PROMPT
    story = STORY.read_text(encoding="utf-8").strip()

    body = {
        "model": "gpt-5.5",
        "reasoning": {"effort": "high"},
        "stream": True,
        "input": [
            {"role": "system", "content": PARENT_SYSTEM_PROMPT},
            {"role": "user", "content": story},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
    }

    print(f"[probe] POST {url}")
    print(f"[probe] model=gpt-5.5  reasoning.effort=high  stream=True")
    print(f"[probe] system_prompt_chars={len(PARENT_SYSTEM_PROMPT)}  story_chars={len(story)}")
    print()

    t0 = time.time()
    last_event_t = t0
    type_counts: dict[str, int] = {}
    out_text_parts: list[str] = []
    reasoning_chars = 0
    big_gap_threshold_s = 5.0

    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=900.0, write=10.0, pool=10.0)) as cli:
        with cli.stream("POST", url, headers=headers, json=body) as resp:
            print(f"[probe] HTTP {resp.status_code}")
            print(f"[probe] headers: cf-ray={resp.headers.get('cf-ray', '-')!s}  "
                  f"content-encoding={resp.headers.get('content-encoding', '-')!s}  "
                  f"transfer-encoding={resp.headers.get('transfer-encoding', '-')!s}")
            print()
            if resp.status_code != 200:
                err_body = resp.read()
                print(f"[probe] ERROR body: {err_body.decode(errors='replace')[:400]}")
                return

            current_event = ""
            for line in resp.iter_lines():
                now = time.time()
                gap = now - last_event_t

                if line.startswith("event:"):
                    current_event = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_str = line.removeprefix("data:").strip()
                    last_event_t = now

                    if not data_str or data_str == "[DONE]":
                        if data_str == "[DONE]":
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  [DONE]")
                        continue

                    try:
                        ev = json.loads(data_str)
                    except json.JSONDecodeError:
                        print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  RAW: {data_str[:120]}")
                        continue

                    etype = ev.get("type", current_event or "?")
                    type_counts[etype] = type_counts.get(etype, 0) + 1

                    if etype == "response.output_text.delta":
                        out_text_parts.append(ev.get("delta", ""))
                        if gap > big_gap_threshold_s or type_counts[etype] == 1:
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  output_text.delta  "
                                  f"({len(out_text_parts)} parts, ~{sum(len(p) for p in out_text_parts)} chars)")
                    elif etype.startswith("response.reasoning"):
                        delta = ev.get("delta") or ev.get("summary", "")
                        if isinstance(delta, str):
                            reasoning_chars += len(delta)
                        if gap > big_gap_threshold_s or type_counts[etype] <= 3:
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  {etype}  ({reasoning_chars} reasoning chars so far)")
                    elif etype in ("response.created", "response.in_progress", "response.completed", "response.failed",
                                   "response.incomplete", "response.output_item.added", "response.output_item.done",
                                   "response.content_part.added", "response.content_part.done",
                                   "response.output_text.done"):
                        print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  {etype}")
                    else:
                        if gap > big_gap_threshold_s:
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  {etype}")

    total_text = "".join(out_text_parts)
    print()
    print(f"=== summary ===")
    print(f"wall:           {time.time()-t0:.1f}s")
    print(f"output_text:    {len(total_text)} chars")
    print(f"reasoning_seen: {reasoning_chars} chars")
    print(f"event types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:50s} x{c}")

    out_path = REPO / "videorlm" / "outputs" / "_refactor_test" / "responses_probe_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"text": total_text, "type_counts": type_counts}, ensure_ascii=False, indent=2))
    print(f"saved to: {out_path}")

    # Try parsing as Skeleton
    print()
    print("=== validate as Skeleton ===")
    try:
        from videorlm.framework.parsers import strip_markdown_fence
        from videorlm.framework.schemas import Skeleton
        body_text = strip_markdown_fence(total_text)
        skel = Skeleton.model_validate_json(body_text)
        print(f"OK Skeleton parsed: shots={len(skel.shots)}")
        portraits = skel.portrait_plan or []
        print(f"portraits: {[p.name for p in portraits]}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
