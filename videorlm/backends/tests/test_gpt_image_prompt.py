from videorlm.backends.media.impl.openai.image.gpt_image_2 import (
    prepare_gpt_image_prompt,
    soften_image_prompt,
)
from videorlm.backends.media.interface.requests import ImageRequest


def test_location_prompt_does_not_inject_a_fashion_woman():
    req = ImageRequest(
        request_id="tower",
        kind="location",
        prompt="黄昏，天庭云海之上，一座正在崩裂的琉璃塔。",
        negative_prompt="现代建筑, 白天, 人物, 文字, 水印",
    )
    text = prepare_gpt_image_prompt(req)
    assert "时尚杂志机位" not in text
    assert "No people" in text
    assert "Avoid also:" in text


def test_fashion_replacements_still_run_on_female_prompts():
    text = soften_image_prompt("一位二十六岁成年东亚女性，性感，低角度仰拍。")
    assert "高级女性美" in text
    assert "时尚杂志机位" in text
