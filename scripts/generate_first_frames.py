#!/usr/bin/env python3
"""Generate high-quality first frames with gpt-image-2 for batch runs."""
from __future__ import annotations

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output", type=Path, default=Path("batch-assets/first-frames"))
    ap.add_argument("--size", default="1536x1024")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    key = os.environ.get("RECA_GPT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("RECA_GPT_API_KEY or OPENAI_API_KEY is required")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    client = OpenAI(api_key=key, base_url=base, max_retries=0)
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    def generate(row: dict) -> str:
        path = args.output / f"{row['id']}.png"
        if path.exists():
            return f"skip {row['id']}"
        response = client.images.generate(
            model="gpt-image-2",
            prompt=row["first_frame_prompt"],
            size=args.size,
            quality="high",
            output_format="png",
        )
        raw = getattr(response.data[0], "b64_json", None)
        if not raw:
            raise RuntimeError(f"gpt-image-2 returned no b64_json for {row['id']}")
        path.write_bytes(base64.b64decode(raw))
        return f"generated {path}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(generate, row) for row in rows]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
