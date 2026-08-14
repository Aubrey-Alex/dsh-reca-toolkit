"""Probe structured output on qwen3.6-max-preview WITH thinking enabled.

Tests both name spellings: 'qwen-3.6-max-preview' and 'qwen3.6-max-preview'.
Goal: find whether ANY structured-output pattern survives thinking mode.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _env_path in (_REPO_ROOT / ".env", _REPO_ROOT / "videorlm" / ".env"):
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                k, v = _line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or (os.environ.get("DASHSCOPE_API_KEYS", "").split(",") or [""])[0].strip()
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = ["qwen3.6-max-preview", "qwen-3.6-max-preview"]
PROMPT = (
    "Analyze this product review and respond with a JSON object containing the "
    "structured fields only: 'Great product, fast shipping, but a bit expensive. "
    "4 out of 5 stars overall.'"
)

from pydantic import BaseModel, Field  # noqa: E402


class ProductReview(BaseModel):
    rating: int = Field(ge=1, le=5)
    sentiment: Literal["positive", "neutral", "negative"]
    key_points: list[str]


JSON_SCHEMA = {
    "name": "ProductReview",
    "schema": {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "minimum": 1, "maximum": 5},
            "sentiment": {"type": "string",
                          "enum": ["positive", "neutral", "negative"]},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["rating", "sentiment", "key_points"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _validate_payload(content) -> ProductReview:
    if isinstance(content, BaseModel):
        return ProductReview.model_validate(content.model_dump())
    if isinstance(content, str):
        content = _strip_think(content)
        content = json.loads(content)
    return ProductReview.model_validate(content)


# Probe a: openai SDK json_schema (thinking ON)
def probe_json_schema(model: str):
    from openai import OpenAI
    c = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120, max_retries=0)
    rsp = c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        temperature=0.0,
        # NO enable_thinking override = use server default (thinking ON for qwen3.6)
    )
    raw = rsp.choices[0].message.content
    return _validate_payload(raw), raw


# Probe b: openai SDK json_object (thinking ON)
def probe_json_object(model: str):
    from openai import OpenAI
    c = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120, max_retries=0)
    rsp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": 'Return ONLY a JSON object: {"rating":int 1-5,'
                        '"sentiment":"positive"|"neutral"|"negative",'
                        '"key_points":[str,...]}.'},
            {"role": "user", "content": PROMPT},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = rsp.choices[0].message.content
    return _validate_payload(raw), raw


# Probe c: LangChain ToolStrategy (thinking ON)
def probe_tool_strategy(model: str):
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_openai import ChatOpenAI
    m = ChatOpenAI(model=model, api_key=API_KEY, base_url=BASE_URL,
                   temperature=0.0, timeout=120)
    agent = create_agent(model=m, tools=[],
                         response_format=ToolStrategy(ProductReview))
    out = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    structured = out.get("structured_response")
    return _validate_payload(structured), structured


# Probe d: LangChain ToolStrategy with auto-recovery (handle_errors on)
def probe_tool_strategy_handled(model: str):
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_openai import ChatOpenAI
    m = ChatOpenAI(model=model, api_key=API_KEY, base_url=BASE_URL,
                   temperature=0.0, timeout=120)
    agent = create_agent(
        model=m, tools=[],
        response_format=ToolStrategy(ProductReview, handle_errors=True),
    )
    out = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    structured = out.get("structured_response")
    return _validate_payload(structured), structured


PROBES = [
    ("a. json_schema   (strict)", probe_json_schema),
    ("b. json_object   (loose)",  probe_json_object),
    ("c. ToolStrategy  (thinking ON)", probe_tool_strategy),
    ("d. ToolStrategy  + handle_errors", probe_tool_strategy_handled),
]


def main():
    grid = {}
    for model in MODELS:
        print("\n" + "#" * 78)
        print(f"# MODEL: {model}")
        print("#" * 78)
        for label, fn in PROBES:
            print(f"\n--- {label} ---")
            try:
                parsed, raw = fn(model)
                print(f"  raw   : {str(raw)[:240]}")
                print(f"  parsed: {parsed.model_dump()}")
                grid[(model, label)] = ("PASS", None)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {str(exc)[:240]}"
                print(f"  FAIL  : {msg}")
                grid[(model, label)] = ("FAIL", msg)

    print("\n" + "=" * 78)
    print("SUMMARY GRID (thinking ENABLED, server default)")
    print("=" * 78)
    print(f"{'probe':<38} | " + " | ".join(f"{m:<22}" for m in MODELS))
    print("-" * 78)
    for label, _ in PROBES:
        cells = []
        for m in MODELS:
            s, _err = grid[(m, label)]
            cells.append(f"{s:<22}")
        print(f"{label:<38} | " + " | ".join(cells))


if __name__ == "__main__":
    if not API_KEY:
        print("[FATAL] no DASHSCOPE_API_KEY")
        sys.exit(2)
    main()
