"""Prompts for the anchor validator agent (vision LLM).

v2 design — 3-layer evaluation (spec self-consistency → image-spec
alignment → image quality) + optional human-in-the-loop override. Failure
type encoded as `[tag]` prefix in `reason`, downstream script branches on
the tag without breaking the strict 7-field schema:

  [design_bug]     anchor.prompt 自身错 (T=0 错位 / 矛盾) → parent revises
  [render_drift]   图跟 spec 不齐 (identity/prop/构图)   → image-edit local
  [low_quality]    artifact / 水印 / 畸形                → re-render fresh
  (no tag)         pass                                  → continue downstream
"""
from __future__ import annotations

from typing import Any


from pathlib import Path as _Path
from videorlm.framework.templates import ChatPromptTemplate as _CPT

_PROMPTS_ROOT = _Path(__file__).resolve().parents[2] / "prompts"
VALIDATOR_SYSTEM_PROMPT: str = _CPT.from_files(
    system=_PROMPTS_ROOT / "validator_anchor" / "system.md",
).format_system()


def format_validator_user_prompt(
    anchor_id: str,
    shot_id: str,
    anchor_prompt: str,
    expected_start_state: str,
    anchor_image_url: str,
    reference_inputs: dict[str, Any],
    expected_end_state: str = "",
    portrait_refs: list[dict[str, Any]] | None = None,
    prop_refs: list[dict[str, Any]] | None = None,
    place_refs: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user turn carrying anchor spec for vision LLM.

    Image itself goes via vision API's image_url channel (not embedded
    here). User prompt explicitly walks the validator through the 3 layers
    so it doesn't skip the spec self-consistency check.
    ``expected_end_state`` is included for the validator to anchor T=0
    vs T>0 disambiguation.
    """
    portrait = reference_inputs.get("portrait", "")
    place = reference_inputs.get("place", "")
    prop = reference_inputs.get("prop", "")
    # 文本描述路径 (无额外图): caller 解析 reference_inputs 的 portrait /
    # prop / place 字段, 把每个 id 对应的 design prompt 文本带进来.
    # validator 通过对比 anchor 图 vs ref design prompt **文本描述**, 判
    # 断 anchor 是否兑现了正确的变体. 比传 portrait PNG 快 4-5×.
    def _block(label: str, refs: list[dict[str, Any]] | None) -> str:
        if not refs:
            return ""
        lines: list[str] = []
        for ref in refs:
            rid = ref.get("id", "?")
            name = ref.get("name", "")
            desc = (ref.get("prompt", "") or "")[:280]
            if not desc:
                lines.append(f"  - id=`{rid}` (name={name!r}): <no design prompt available>")
            else:
                lines.append(
                    f"  - id=`{rid}` (name={name!r}):\n"
                    f"    design prompt: {desc}"
                )
        return f"\n[{label} — 用这段文本描述对比 anchor 实际渲染]\n" + "\n".join(lines) + "\n"

    refs_block = (
        _block("portrait refs (角色)", portrait_refs)
        + _block("prop refs (道具)", prop_refs)
        + _block("place refs (场景)", place_refs)
    )
    body = f"""
本次任务: 验证 anchor `{anchor_id}` (属于 shot `{shot_id}`)。
"""
    body += f"""
请严格按 system_prompt 的 3 层工作流程:
1. **第一层 (设计自洽)** — 不看图, 先比较下面的 anchor.prompt vs shot.start_state 时间点是否一致 (T=0 vs T>0)
2. **第二层 (图-spec 对齐)** — 看图, 按 5 维度加权打分 (identity 0.30 / prop 0.25 / composition 0.20 / color 0.15 / T=0 静态 0.10)
3. **第三层 (图像质量)** — 检查 artifact / 水印 / 畸形

[anchor.prompt — 渲染时的 i2i 指令]
{anchor_prompt}

[shot.start_state — 应该是同一帧的剧本简述 (T=0 ground truth)]
{expected_start_state}

[shot.end_state — 末帧, 帮你区分 T=0 vs T>0]
{expected_end_state}

[reference_inputs — 引用资产 ID, identity 必须保留]
- portrait: {portrait}
- place: {place}
- prop: {prop}
{refs_block}
[anchor 渲染图 URL]
{anchor_image_url}
(仅 anchor 这一张图经 vision API image_url 通道附带; portrait / prop / place 等 ref
不传图, 只走上面 [refs] 段的 design prompt 文本描述)

请输出严格 7 字段 JSON judgment, reason 以 `[design_bug]` / `[render_drift]` / `[low_quality]` 开头, 或 pass 时无 tag。包在 ```json ... ``` fence 中。
"""
    return body


__all__ = ["VALIDATOR_SYSTEM_PROMPT", "format_validator_user_prompt"]
