from __future__ import annotations

import unittest
from unittest.mock import patch

from videorlm.framework.pipeline import _build_image_dag, _render_image_dag


class PipelineInputTests(unittest.TestCase):
    def test_preloaded_anchor_is_not_rendered_again(self) -> None:
        render_plan = {
            "preloaded_assets": {"anchor": "/tmp/user-first-frame.png"},
            "boundary_anchors": [],
        }
        deps = {"portrait": set(), "anchor": {"portrait"}}
        nodes = {
            "portrait": ({"request_id": "portrait", "references": []}, "portrait"),
            "anchor": ({"request_id": "anchor", "references": []}, "anchor_image"),
        }
        with patch("videorlm.framework.pipeline._build_image_dag", return_value=(deps, nodes)), \
             patch("videorlm.framework.pipeline._dispatch_one_image", return_value="https://image/portrait.png") as dispatch:
            result = _render_image_dag(render_plan, max_workers=1)
        self.assertEqual(result["anchor"], "/tmp/user-first-frame.png")
        self.assertEqual(result["portrait"], "https://image/portrait.png")
        self.assertEqual(dispatch.call_count, 1)

    def test_anchor_edit_chain_is_a_dag_dependency(self) -> None:
        render_plan = {
            "portrait_plan": {},
            "location_plan": {},
            "prop_plan": {},
            "boundary_anchors": [
                {"id": "a01", "image_request": {"references": []}},
                {"id": "a02", "image_request": {
                    "references": [{"role": "source", "asset_id": "a01", "url": ""}],
                    "render_kind": "image_edit", "source_anchor": "a01",
                }},
            ],
        }
        deps, nodes = _build_image_dag(render_plan)
        self.assertEqual(deps["a01"], set())
        self.assertEqual(deps["a02"], {"a01"})
        self.assertEqual(nodes["a02"][1], "image_edit")


if __name__ == "__main__":
    unittest.main()
