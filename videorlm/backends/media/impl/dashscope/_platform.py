"""DashScope vendor-shared helpers.

Centralizes the auth / key-pool / submit / endpoint plumbing reused by every
dashscope-backed image and video model so the per-model files stay focused
on capability declaration + payload shaping.
"""
from __future__ import annotations

from ...._common.dashscope_sdk import classify_dashscope_response
from ...._common.platforms import get_platform, with_key


_PLATFORM = get_platform("dashscope")


def dashscope_key_pool():
    return _PLATFORM.key_pool()


def platform():
    return _PLATFORM


def submit_with_key(call):
    """DashScope-flavored wrapper around the generic ``with_key`` middleware.

    Kept as a 1-line shim so existing
    ``submit_with_key(lambda api_key: VideoSynthesis.async_call(...))``
    call sites in ``image/_primitives.py`` / ``video/_primitives.py``
    stay path-stable.
    """
    return with_key("dashscope", call, classify_response=classify_dashscope_response)
