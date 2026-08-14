from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gateway.artifacts import public_manifest
from gateway.recovery import recover_unfinished_runs
from gateway.schemas import normalize_run_config
from videorlm.integrations.director.runtime import (
    write_artifact_manifest,
    write_audit_report,
    write_run_report,
    write_state,
)


class GatewayContractTests(unittest.TestCase):
    def test_run_config_normalizes_product_fields(self) -> None:
        config = normalize_run_config({
            "duration": 30,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "enable_audit": True,
        })
        self.assertEqual(config.duration_s, 30)
        self.assertEqual(config.style, "cinematic")
        self.assertTrue(config.enable_audit)

    def test_recovery_marks_only_active_runs_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            tmp_path = Path(value)
            active = tmp_path / "active"
            active.mkdir()
            (active / "state.json").write_text(json.dumps({"run_id": "active", "state": "running"}))
            done = tmp_path / "done"
            done.mkdir()
            (done / "state.json").write_text(json.dumps({"run_id": "done", "state": "succeeded"}))
            self.assertEqual(recover_unfinished_runs(tmp_path), ["active"])
            self.assertEqual(json.loads((active / "state.json").read_text())["state"], "interrupted")
            self.assertEqual(json.loads((done / "state.json").read_text())["state"], "succeeded")

    def test_manifest_publishes_urls_without_reading_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            tmp_path = Path(value)
            run = tmp_path / "run"
            run.mkdir()
            (run / "final.mp4").write_bytes(b"mp4")
            manifest = public_manifest(tmp_path, "abc123", "http://localhost:8787")
            final = next(item for item in manifest["artifacts"] if item["kind"] == "final_video")
            self.assertEqual(final["status"], "ready")
            self.assertTrue(final["url"].endswith("/v1/runs/abc123/artifacts/run/final.mp4"))

    def test_reca_runtime_writes_atomic_state_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            tmp_path = Path(value)
            (tmp_path / "run" / "final.mp4").parent.mkdir(parents=True)
            (tmp_path / "run" / "final.mp4").write_bytes(b"mp4")
            write_state(
                tmp_path,
                stage="succeeded",
                state="succeeded",
                audit_state="audit_skipped",
                video_state="complete",
                progress=1.0,
            )
            write_audit_report(tmp_path, state="audit_skipped")
            write_run_report(tmp_path, state="succeeded")
            manifest_path = write_artifact_manifest(tmp_path, run_id="run123")
            state = json.loads((tmp_path / "run" / "reca_state.json").read_text())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(state["stage"], "succeeded")
            self.assertEqual(state["video_state"], "complete")
            self.assertEqual(manifest["run_id"], "run123")
            kinds = {item["kind"]: item for item in manifest["artifacts"]}
            self.assertEqual(kinds["audit"]["status"], "skipped")
            self.assertEqual(kinds["run_report"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
