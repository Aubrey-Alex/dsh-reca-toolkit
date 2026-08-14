"""Probe /v1/messages (Anthropic Messages API ingress).

Exercises Anthropic-style streaming against ``OPENAI_BASE_URL`` (which
may be openai.com directly or a gateway in front of it).

Body format = Anthropic Messages API:
    {model, system, messages, max_tokens, stream, thinking: {type, budget_tokens}}

Streaming events:
    event: message_start    | data: {...}
    event: content_block_delta | data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
    event: message_delta    | data: {...}
    event: message_stop     | data: {...}
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
    url = base_url + "/messages"
    if not api_key:
        raise SystemExit("missing key (set $OPENAI_API_KEY)")

    from videorlm.framework.shot_planner.prompts import PARENT_SYSTEM_PROMPT
    story = STORY.read_text(encoding="utf-8").strip()

    body = {
        "model": "gpt-5.5",
        "system": PARENT_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": story}],
        "max_tokens": 32000,
        "stream": True,
        "thinking": {"type": "enabled", "budget_tokens": 16000},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "anthropic-version": "2023-06-01",
        "Accept-Encoding": "identity",
    }

    print(f"[probe-msg] POST {url}")
    print(f"[probe-msg] model=gpt-5.5  thinking.budget=16000  stream=True")
    print(f"[probe-msg] system={len(PARENT_SYSTEM_PROMPT)}c  user={len(story)}c")
    print()

    t0 = time.time()
    last_t = t0
    type_counts: dict[str, int] = {}
    text_parts: list[str] = []
    thinking_chars = 0
    big_gap = 5.0

    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=900.0, write=10.0, pool=10.0)) as cli:
        with cli.stream("POST", url, headers=headers, json=body) as resp:
            print(f"[probe-msg] HTTP {resp.status_code}")
            print(f"[probe-msg] cf-ray={resp.headers.get('cf-ray', '-')!s}  "
                  f"ce={resp.headers.get('content-encoding', '-')!s}  "
                  f"te={resp.headers.get('transfer-encoding', '-')!s}")
            print()
            if resp.status_code != 200:
                err = resp.read()
                print(f"[probe-msg] ERROR body: {err.decode(errors='replace')[:500]}")
                return

            current_event = ""
            for line in resp.iter_lines():
                now = time.time()
                gap = now - last_t

                if line.startswith("event:"):
                    current_event = line.removeprefix("event:").strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line.removeprefix("data:").strip()
                last_t = now
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    ev = json.loads(data_str)
                except json.JSONDecodeError:
                    print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  RAW: {data_str[:100]}")
                    continue
                etype = ev.get("type", current_event or "?")
                type_counts[etype] = type_counts.get(etype, 0) + 1

                if etype == "content_block_delta":
                    delta = ev.get("delta") or {}
                    dtype = delta.get("type", "")
                    if dtype == "text_delta":
                        text_parts.append(delta.get("text", ""))
                        if gap > big_gap or type_counts[etype] == 1:
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  text_delta  "
                                  f"(~{sum(len(p) for p in text_parts)} chars)")
                    elif dtype == "thinking_delta":
                        thinking_chars += len(delta.get("thinking", ""))
                        if gap > big_gap or type_counts[etype] <= 3:
                            print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  thinking_delta  "
                                  f"({thinking_chars} thinking chars)")
                elif etype in ("message_start", "message_delta", "message_stop",
                               "content_block_start", "content_block_stop"):
                    print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  {etype}")
                else:
                    if gap > big_gap:
                        print(f"[+{now-t0:7.1f}s] gap={gap:5.1f}s  {etype}")

    total = "".join(text_parts)
    print()
    print(f"=== summary ===")
    print(f"wall:           {time.time()-t0:.1f}s")
    print(f"text:           {len(total)} chars")
    print(f"thinking_seen:  {thinking_chars} chars")
    print(f"event types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:50s} x{c}")

    out = REPO / "videorlm" / "outputs" / "_refactor_test" / "messages_probe_output.json"
    out.write_text(json.dumps({"text": total, "type_counts": type_counts}, ensure_ascii=False, indent=2))

    print()
    print("=== validate as Skeleton ===")
    try:
        from videorlm.framework.parsers import strip_markdown_fence
        from videorlm.framework.schemas import Skeleton
        skel = Skeleton.model_validate_json(strip_markdown_fence(total))
        print(f"OK shots={len(skel.shots)}  portraits={[p.name for p in (skel.portrait_plan or [])]}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
