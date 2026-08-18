from __future__ import annotations

import unittest

from videorlm.framework._scripts._smoke import _apply_director_inputs


class DirectorInputTests(unittest.TestCase):
    def test_first_frame_preloads_first_anchor_and_refs_are_forwarded(self) -> None:
        render_plan = {
            "boundary_anchors": [
                {"id": "a01_start", "image_request": {"references": []}},
            ],
            "segments": {},
        }
        _apply_director_inputs(render_plan, {
            "first_frame": {"path": "/tmp/first.png"},
            "reference_images": [
                {"path": "/tmp/hero.png", "role": "character", "name": "hero"},
            ],
        })
        self.assertEqual(render_plan["preloaded_assets"]["a01_start"], "/tmp/first.png")
        self.assertEqual(render_plan["protected_anchor_ids"], ["a01_start"])
        self.assertEqual(render_plan["provided_reference_images"][0]["name"], "hero")
        self.assertEqual(
            render_plan["boundary_anchors"][0]["image_request"]["references"][0]["url"],
            "/tmp/hero.png",
        )

    def test_first_frame_makes_later_anchors_serial_image_edits(self) -> None:
        render_plan = {
            "boundary_anchors": [
                {"id": "a01", "image_request": {"references": []}},
                {"id": "a02", "image_request": {"references": []}},
                {"id": "a03", "image_request": {"references": []}},
            ],
        }
        _apply_director_inputs(render_plan, {"first_frame": {"path": "/tmp/first.png"}})

        self.assertEqual(render_plan["canonical_reference_image"], "/tmp/first.png")
        self.assertEqual(render_plan["preloaded_assets"], {"a01": "/tmp/first.png"})
        a02 = render_plan["boundary_anchors"][1]["image_request"]
        a03 = render_plan["boundary_anchors"][2]["image_request"]
        self.assertEqual(a02["render_kind"], "image_edit")
        self.assertEqual(a02["source_anchor"], "a01")
        self.assertEqual(a03["source_anchor"], "a02")
        self.assertEqual(a02["references"][0], {"role": "source", "url": "", "asset_id": "a01"})


if __name__ == "__main__":
    unittest.main()
