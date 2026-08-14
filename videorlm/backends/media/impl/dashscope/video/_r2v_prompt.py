"""Shared R2V soft-target prompt templates.

Two backends, two templates:

  - ``R2V_PROMPT_TEMPLATE`` (wan2.7-r2v):
      wan2.7-r2v takes a real ``first_frame`` slot — the SDK pins frame 0
      hard, so prompt only needs to talk about the SOFT last anchor:
      "use the reference image as the target next-shot state".

  - ``HAPPYHORSE_R2V_PROMPT_TEMPLATE`` (happyhorse-1.0-r2v):
      happyhorse-r2v has NO first_frame slot — the API only takes
      reference_image[s]. We put first_url into reference_image[0] but
      the model is free to ignore it. To soft-anchor the start frame we
      explicitly tell the model: "the first reference image IS the start
      frame, begin from it". This is best-effort soft anchoring (caller
      already saw `flf2v_mode="all_soft"`).

Used by both r2v backends so the prefix is applied consistently.
"""
from __future__ import annotations


R2V_PROMPT_TEMPLATE = (
    "Start exactly from the first frame. "
    "Use the reference image as the target next-shot state. "
    "Move naturally toward that state with continuous camera motion. "
    "Preserve subject identity, wardrobe, lighting, spatial layout, and color palette. "
    "No hard reset, no identity change, no extra people, no text, no watermark."
)


# happyhorse-r2v 没有 first_frame slot,只能在 prompt 显式声明 reference[0] 是首帧。
HAPPYHORSE_R2V_PROMPT_TEMPLATE = (
    "把第一张参考图作为视频首帧,从该画面开始,后续参考图作为风格/身份软目标。"
)


STORYBOARD_R2V_PROMPT_TEMPLATE = (
    "Use the reference storyboard only as temporal guidance, not as a visible layout. "
    "hard cut only, no dissolve, no morphing, no double exposure, no ghost overlay. "
    "Do not draw panel borders, labels, or split-screen layout."
)


def prefix_r2v(stem: str, prompt: str) -> str:
    """Prepend a soft-target instruction to a prompt body."""
    return f"{stem}\n\n{prompt or ''}".strip()
