"""Sanity checks for the optional DSH helper script."""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start-dsh.sh"


class StartDshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = SCRIPT.read_text(encoding="utf-8")

    def test_script_parses(self) -> None:
        subprocess.check_call(["bash", "-n", str(SCRIPT)])

    def test_installs_plugin_then_opens_web(self) -> None:
        self.assertIn("install_dsh_plugin.sh", self.text)
        self.assertIn("dsh web", self.text)
        self.assertNotIn("gateway.server", self.text)

    def test_does_not_kill_foreign_processes(self) -> None:
        self.assertNotIn("pkill", self.text)
        self.assertNotIn("killall", self.text)


if __name__ == "__main__":
    unittest.main()
