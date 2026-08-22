import tempfile
import unittest
from pathlib import Path

from gateway.jobs import allocate_run_id, slugify_run_id, story_run_slug


class RunIdSlugTests(unittest.TestCase):
    def test_story_slug_uses_first_line(self) -> None:
        self.assertEqual(slugify_run_id("Nezha Fire Pagoda"), "nezha_fire_pagoda")
        self.assertEqual(story_run_slug("Cliff gaze at sunrise\nMore story."), "cliff_gaze_at_sunrise")

    def test_allocate_increments_existing_folder(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "film").mkdir()
            self.assertEqual(allocate_run_id(root, "film"), "film_2")


if __name__ == "__main__":
    unittest.main()
