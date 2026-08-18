from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.artifacts import public_manifest
from gateway.recovery import recover_unfinished_runs
from gateway.jobs import JobManager
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

    def test_gateway_stages_optional_first_frame_and_references(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            source = root / "source.png"
            source.write_bytes(b"png-bytes")
            job = root / "run123"
            job.mkdir()
            manager = JobManager(root=Path.cwd(), runs_root=root / "runs")
            manifest = manager._stage_input_assets(
                job,
                first_frame=str(source),
                reference_image_urls=[str(source)],
                reference_images=[{"path": str(source), "role": "character", "name": "hero"}],
            )
            self.assertTrue(Path(manifest["first_frame"]["path"]).is_file())
            self.assertEqual(manifest["reference_images"][0]["role"], "character")
            self.assertEqual(len(manifest["reference_images"]), 2)
            self.assertTrue((job / "input_manifest.json").is_file())

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
            recovered = json.loads((active / "state.json").read_text())
            self.assertEqual(recovered["state"], "interrupted")
            self.assertEqual(recovered["gateway_state"], "interrupted")
            self.assertEqual(recovered["stage"], "interrupted")
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

    def test_reca_runtime_marks_enabled_audit_ready(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            tmp_path = Path(value)
            (tmp_path / "run").mkdir()
            write_audit_report(tmp_path, state="audited", details={"pass": True})
            manifest_path = write_artifact_manifest(tmp_path, run_id="audited123")
            manifest = json.loads(manifest_path.read_text())
            kinds = {item["kind"]: item for item in manifest["artifacts"]}
            self.assertEqual(kinds["audit"]["status"], "ready")

    def test_gateway_restart_marks_run_interrupted_and_resume_reuses_run(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            runs_root = Path(value)
            run_id = "a" * 12
            run = runs_root / run_id
            run.mkdir()
            (run / "state.json").write_text(json.dumps({
                "run_id": run_id,
                "state": "running",
                "gateway_state": "running",
                "stage": "segments",
            }))
            (run / "request.json").write_text(json.dumps({
                "story": "resume this story",
                "options": {"backend": "wan", "enable_audit": False},
            }))

            manager = JobManager(root=Path.cwd(), runs_root=runs_root)
            recovered = json.loads((run / "state.json").read_text())
            self.assertEqual(recovered["gateway_state"], "interrupted")
            with patch.object(manager, "start", return_value={"run_id": run_id}) as start:
                self.assertEqual(manager.resume(run_id), {"run_id": run_id})
                request = start.call_args.args[0]
                self.assertEqual(request["options"]["resume_run_id"], run_id)


if __name__ == "__main__":
    unittest.main()
