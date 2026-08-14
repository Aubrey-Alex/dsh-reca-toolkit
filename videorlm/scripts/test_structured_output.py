"""Probe whether DashScope qwen3.6-max-preview supports LangChain v1 structured output.

Reference: https://langchain-doc.cn/v1/python/langchain/structured-output.html

Runs four independent probes and prints a PASS/FAIL summary:
  1. OpenAI SDK: response_format={"type":"json_schema", ...}     (raw native)
  2. OpenAI SDK: response_format={"type":"json_object"}            (loose JSON mode)
  3. LangChain v1 create_agent(..., response_format=ToolStrategy(Schema))
  4. LangChain v1 create_agent(..., response_format=ProviderStrategy(Schema))
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Literal

# --- env bootstrap: load .env so DASHSCOPE_API_KEY is available --------------
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
    or os.environ.get("dashscope_new_api_key")
    or (os.environ.get("DASHSCOPE_API_KEYS", "").split(",") or [""])[0].strip()
)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = os.environ.get("PROBE_MODEL", "qwen3.6-max-preview")

if not API_KEY:
    print("[FATAL] no DASHSCOPE_API_KEY found in env or .env")
    sys.exit(2)

print(f"[setup] model={MODEL}  base_url={BASE_URL}  key=***{API_KEY[-4:]}")

PROMPT = (
    "Analyze this product review and respond as a JSON object with the structured "
    "fields only: 'Great product, fast shipping, but a bit expensive. 4 out of 5 "
    "stars overall.'"
)

# DashScope qwen3.x defaults to thinking mode; that blocks tool_choice=required
# and prepends <think>...</think>. Disable it for structured-output probes.
NO_THINK = {"enable_thinking": False}

# Shared schema (Pydantic + JSON schema literal)
from pydantic import BaseModel, Field  # noqa: E402


class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int = Field(description="Product rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment"
    )
    key_points: list[str] = Field(
        description="Key points, each 1-3 lowercase words"
    )


JSON_SCHEMA = {
    "name": "ProductReview",
    "schema": {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "minimum": 1, "maximum": 5},
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["rating", "sentiment", "key_points"],
        "additionalProperties": False,
    },
    "strict": True,
}


def _validate(payload) -> ProductReview:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, BaseModel):
        return ProductReview.model_validate(payload.model_dump())
    return ProductReview.model_validate(payload)


# --- Probe 1: raw OpenAI SDK + response_format=json_schema -------------------
def probe_json_schema():
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=0)
    rsp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        temperature=0.0,
        extra_body=NO_THINK,
    )
    content = rsp.choices[0].message.content
    print(f"[raw   ] {content}")  # dashscope sometimes silently ignores schema
    parsed = _validate(content)
    return parsed, content


# --- Probe 2: raw OpenAI SDK + response_format=json_object -------------------
def probe_json_object():
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=0)
    schema_hint = (
        "Return ONLY a JSON object with keys "
        '{"rating": int 1-5, "sentiment": "positive"|"neutral"|"negative", '
        '"key_points": [str, ...]}.'
    )
    rsp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": schema_hint},
            {"role": "user", "content": PROMPT},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        extra_body=NO_THINK,
    )
    content = rsp.choices[0].message.content
    # qwen3.x may emit <think>...</think> — strip it
    import re
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    parsed = _validate(content)
    return parsed, content


# --- Probe 3: LangChain v1 create_agent + ToolStrategy -----------------------
def probe_langchain_tool_strategy():
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
        timeout=60,
        extra_body=NO_THINK,
    )
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(ProductReview),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    structured = result.get("structured_response")
    parsed = _validate(structured)
    return parsed, structured


# --- Probe 4: LangChain v1 create_agent + ProviderStrategy -------------------
def probe_langchain_provider_strategy():
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ProviderStrategy
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.0,
        timeout=60,
        extra_body=NO_THINK,
    )
    agent = create_agent(
        model=model,
        tools=[],
        response_format=ProviderStrategy(ProductReview),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": PROMPT}]})
    structured = result.get("structured_response")
    parsed = _validate(structured)
    return parsed, structured


PROBES = [
    ("1. OpenAI-SDK  response_format=json_schema  (strict native)",
     probe_json_schema),
    ("2. OpenAI-SDK  response_format=json_object  (loose JSON mode)",
     probe_json_object),
    ("3. LangChain   create_agent + ToolStrategy",
     probe_langchain_tool_strategy),
    ("4. LangChain   create_agent + ProviderStrategy",
     probe_langchain_provider_strategy),
]


def main():
    results = []
    for label, fn in PROBES:
        print("\n" + "=" * 78)
        print(f"[probe] {label}")
        print("=" * 78)
        try:
            parsed, raw = fn()
            print(f"[raw   ] {str(raw)[:300]}")
            print(f"[parsed] {parsed.model_dump()}")
            print(f"[PASS  ] {label}")
            results.append((label, "PASS", None))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=3)
            short = f"{type(exc).__name__}: {str(exc)[:240]}"
            print(f"[FAIL  ] {short}")
            print(tb)
            results.append((label, "FAIL", short))

    print("\n" + "#" * 78)
    print(f"# Summary for model {MODEL}")
    print("#" * 78)
    for label, status, err in results:
        line = f"  [{status}] {label}"
        if err:
            line += f"  -- {err}"
        print(line)
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} probes passed.")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
