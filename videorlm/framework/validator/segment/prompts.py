"""Prompts for the segment (post-render) visual validator agent.

3-axis rubric on the rendered segment MP4 (the model samples frames over
the full timeline — NOT a single tail PNG):
  A. 美观 (aesthetic / cinematic)        — composition, lighting, clarity, no artifact
  B. 符合全局 (global alignment)         — segment.end_state + shot.story_goal + visual_intent
  C. 动作连贯一致性 (action coherence)   — motif preserved + identity stable + physical contact

8-field strict JSON out (segment_id / aesthetic_score / global_alignment_score /
action_consistency_score / overall_score / pass_or_not / reason / repair_action / edit_prompt).
Mapping of failure → repair_action (paper §5.3 4-action menu, framework implements 3):

  weakest axis = aesthetic       → action = "RegenerateUnit"  (re-render, often artifact issue)
  weakest axis = global_alignment → action = "RepackPrompt"   (rewrite prompt to anchor goal/end)
  weakest axis = action_consistency → action = "RepackPrompt"  (append motif/identity clause)
  if identity completely lost      → action = "ReanchorState"  (rare)
  pass                              → action = ""              (no repair)
"""
from __future__ import annotations

from typing import Any


from pathlib import Path as _Path
from videorlm.framework.templates import ChatPromptTemplate as _CPT

_PROMPTS_ROOT = _Path(__file__).resolve().parents[2] / "prompts"
SEGMENT_VALIDATOR_SYSTEM_PROMPT: str = _CPT.from_files(
    system=_PROMPTS_ROOT / "validator_segment" / "system.md",
).format_system()


def format_segment_validator_user_prompt(
    segment_id: str,
    segment_prompt: str,
    segment_end_state: str,
    shot_start_state: str,
    segment_video_url: str,
    portrait_urls: list[str],
    *,
    shot_story_goal: str = "",
    shot_visual_intent: str = "",
) -> str:
    """Build the user turn for the segment vision validator.

    The mp4 + portrait refs are passed via the multimodal API:
      ``prompt_with_video(text, segment_video_url, portrait_urls)``
    The video carries the FULL timeline (model samples its own frames),
    portraits are still image refs for the identity axis.

    The user prompt covers the 3 axes laid out in the system prompt:
      A. 美观 (aesthetic) — composition/clarity over the full video
      B. 符合全局 (global alignment) — story_goal + end_state + visual_intent
         (end_state checks the video's last 1-2 seconds)
      C. 动作连贯一致性 (consistency) — needs the full video to judge
         physical contact / motif / identity continuity over time
    """
    # Portrait reference URLs are passed to the model via the multimodal
    # ``image`` content channel — they DO NOT belong in the prompt text.
    # Earlier versions inlined the URLs here, which silently embedded full
    # ``data:image/png;base64,...`` blobs (~1.5 MB / portrait) into the
    # user message and overflowed qwen3-vl-plus's 260 096-token input cap
    # whenever any portrait was a data URI. Now we just announce how many
    # identity refs are attached; the model already has them as images.
    n_portraits = sum(1 for u in (portrait_urls or []) if u)
    if n_portraits:
        portrait_summary = (
            f"  ({n_portraits} 张 portrait 参考图通过 image_url 通道附带 — "
            f"image[0..{n_portraits - 1}] 即 identity reference, 按顺序对应)"
        )
    else:
        portrait_summary = "  (no portrait refs available; identity axis falls back to internal coherence check)"
    return f"""
本次任务: 验证 segment `{segment_id}` 的**整段视频** (mp4 时间轴), 给 3 个 axis
各打分 (aesthetic / global_alignment / action_consistency)。

[segment.prompt — R2V/I2V 渲染时的指令]
{segment_prompt}

[segment.end_state — 视频末点(最后 1-2 秒)应该出现的画面快照]
{segment_end_state}

[shot.story_goal — 该 shot 在故事里的功能/意义 (global_alignment axis 重点参考)]
{shot_story_goal or "(无 story_goal,跳过 story_goal 维度)"}

[shot.start_state — 该 shot 在 T=0 的剧本简述 (motif 的来源, action_consistency axis 重点参考)]
{shot_start_state}

[shot.visual_intent — 镜头运动/节奏/风格/氛围 (global_alignment + aesthetic axis 参考)]
{shot_visual_intent or "(无 visual_intent, 仅按一般电影感判)"}

[portrait refs — identity 参考图 (action_consistency.identity 子项参考)]
{portrait_summary}

[segment video — 实际渲出来的 mp4]
  (通过 video 通道附带, 你看完整时间轴)

★ 评 action_consistency 时**用电影合理性判, 不是绝对硬规则**:

1. 先读懂 segment.prompt 的**叙事节拍** (例: 紧逼 → 爆开 → 分立),不要逐字对照
2. 看视频是否**电影化地兑现**这个叙事弧:
   - 节拍按顺序出现就算合理 (顺序对 + 大致画面对得上 prompt)
   - 焦点切换是正常电影手法 (近景 → 拉远 → 跟随等),不扣分
   - 主角**在被聚焦时**身份一致 (不是真人脸 / 重影 / 跟 portrait 完全不同人)
   - 关键 motif (神目 / 头冠 / 武器) 在被聚焦时认得出来
   - 物理无明显违和 (穿模 / 飘浮 / 反重力)
3. 不该做的事:
   - ❌ **不要编 timestamp** (T=2.0s 这种) — prompt 没写不要假装
   - ❌ 不要用"全程必须 X"做硬性要求 — 除非 prompt 明确写了"全程 / 始终"
   - ❌ 不要把"镜头焦点切换"当成"角色消失"扣分

真正该 fail 的情况:叙事节拍**缺失** (该有的没出现) / identity 整段漂 / 严重穿模。

请输出严格 9 字段 JSON judgment, 包在 ```json ... ``` fence 中。
"""


__all__ = [
    "SEGMENT_VALIDATOR_SYSTEM_PROMPT",
    "format_segment_validator_user_prompt",
]
