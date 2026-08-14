"""Anthropic-Messages-format agent over an OpenAI-compatible gateway.

Why this is a separate agent (not a subclass of ``OpenAICompatibleAgent``):
the wire format is Anthropic's, not OpenAI's. We POST to ``/v1/messages``
with ``{system, messages, max_tokens, stream, thinking}`` and read SSE
events typed ``message_start / content_block_delta / message_delta /
message_stop`` — completely different schema from chat.completions.

For gpt-5 / gpt-5.5 reasoning models on a translation gateway this path
avoids two failure modes of the ``/v1/chat/completions`` route:

  1. ``/chat/completions`` translation drops the reasoning event stream,
     so the client sees long mid-stream silence and the gateway's
     ``stream_data_interval_timeout`` fires → truncated reply.

  2. With ``thinking: {type: enabled}``, the Anthropic SSE explicitly
     interleaves ``content_block_delta`` (thinking_delta) and a second
     ``content_block`` for text — there is no silence at any point.

Empirically (probe_messages_api.py, 2026-05-16 ex01 story):
  - one-shot success, 232s wall, 22372-char JSON
  - shots=20, portraits include both 巨猿孙悟空 and 法天象地杨戬
  - vs ``/chat/completions`` gpt-5.5: 2 attempts, 438s wall, 17 shots,
    2 portraits (no transformation forms)
"""
from .agent import OpenAIMessagesAgent
from .config import OpenAIMessagesConfig

__all__ = ["OpenAIMessagesAgent", "OpenAIMessagesConfig"]
