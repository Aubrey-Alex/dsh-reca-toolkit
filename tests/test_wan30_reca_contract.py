from __future__ import annotations

import unittest
from unittest.mock import patch

from videorlm.backends.media.impl.dashscope.video import wan30
from videorlm.backends.media.interface.requests import BridgeRequest, SegmentRequest


class Wan30ReCAContractTests(unittest.TestCase):
    @staticmethod
    def _capture_generate(calls: list[dict]):
        def fake_generate(**kwargs):
            calls.append(kwargs)
            return "https://provider.example/result.mp4"

        return fake_generate

    def test_r2v_uses_happyhorse_style_reference_images(self) -> None:
        calls: list[dict] = []
        request = SegmentRequest(
            request_id="segment-r2v",
            prompt="continue the planned action",
            first_url="https://assets.example/previous-tail.png",
            mode="r2v",
            reference_image_urls=tuple(
                f"https://assets.example/reference-{index}.png" for index in range(5)
            ),
            output_path="/tmp/segment-r2v.mp4",
        )

        with patch.object(wan30, "_generate", side_effect=self._capture_generate(calls)):
            wan30.Wan30VideoBackend().render_segment(request)

        self.assertEqual(
            calls[0]["media"],
            [
                {"type": "reference_image", "url": request.first_url},
                *[
                    {"type": "reference_image", "url": url}
                    for url in request.reference_image_urls[:3]
                ],
            ],
        )
        self.assertIn("第一张参考图作为视频首帧", calls[0]["prompt"])

    def test_i2v_uses_only_the_original_first_frame(self) -> None:
        calls: list[dict] = []
        request = SegmentRequest(
            request_id="segment-i2v",
            prompt="continue the planned action",
            first_url="https://assets.example/previous-tail.png",
            mode="i2v",
            reference_image_urls=("https://assets.example/ignored.png",),
            output_path="/tmp/segment-i2v.mp4",
        )

        with patch.object(wan30, "_generate", side_effect=self._capture_generate(calls)):
            wan30.Wan30VideoBackend().render_segment(request)

        self.assertEqual(calls[0]["media"], [{"type": "reference_image", "url": request.first_url}])
        self.assertEqual(calls[0]["prompt"], request.prompt)

    def test_bridge_preserves_original_first_and_last_frames(self) -> None:
        calls: list[dict] = []
        request = BridgeRequest(
            request_id="bridge",
            prompt="transition continuously",
            first_url="https://assets.example/previous-shot-tail.png",
            last_url="https://assets.example/next-shot-anchor.png",
            output_path="/tmp/bridge.mp4",
        )

        with patch.object(wan30, "_generate", side_effect=self._capture_generate(calls)):
            wan30.Wan30VideoBackend().render_bridge(request)

        self.assertEqual(
            calls[0]["media"],
            [
                {"type": "first_frame", "url": request.first_url},
                {"type": "last_frame", "url": request.last_url},
            ],
        )


if __name__ == "__main__":
    unittest.main()
