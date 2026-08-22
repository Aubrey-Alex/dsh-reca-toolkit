from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gateway.server import GatewayServer


class FakeManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[str, str]] = []
        self.status_value = {
            "run_id": "abc123",
            "state": "running",
            "gateway_state": "running",
            "artifact_manifest": {"run_id": "abc123", "artifacts": []},
        }
        (root / "final.mp4").write_bytes(b"video")

    def list_runs(self):
        return [self.status_value]

    def status(self, run_id):
        return self.status_value if run_id == "abc123" else None

    def start(self, body):
        self.calls.append(("start", body["story"]))
        return self.status_value

    def cancel(self, run_id):
        self.calls.append(("cancel", run_id))
        return self.status_value if run_id == "abc123" else None

    def resume(self, run_id):
        self.calls.append(("resume", run_id))
        return self.status_value if run_id == "abc123" else None

    def events(self, run_id):
        return [] if run_id == "abc123" else None

    def artifact_path(self, run_id, relative):
        if run_id == "abc123" and relative == "final.mp4":
            return self.root / "final.mp4"
        return None


class GatewayHttpLifecycleTests(unittest.TestCase):
    def test_http_lifecycle_routes(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            manager = FakeManager(Path(value))
            server = GatewayServer(("127.0.0.1", 0), manager)  # type: ignore[arg-type]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urlopen(base + "/health") as response:
                    self.assertEqual(response.status, 200)
                    health = json.loads(response.read())
                    self.assertTrue(health["ok"])
                    self.assertEqual(health["service"], "reca-gateway")
                    self.assertTrue(health["repo_root"])

                request = Request(
                    base + "/v1/runs",
                    data=json.dumps({"story": "a short story"}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request) as response:
                    self.assertEqual(response.status, 202)
                self.assertIn(("start", "a short story"), manager.calls)

                with urlopen(base + "/v1/runs/abc123") as response:
                    self.assertEqual(json.loads(response.read())["gateway_state"], "running")

                for action in ("cancel", "resume"):
                    request = Request(
                        f"{base}/v1/runs/abc123/{action}",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                    )
                    with urlopen(request) as response:
                        self.assertEqual(response.status, 200 if action == "cancel" else 202)
                self.assertIn(("cancel", "abc123"), manager.calls)
                self.assertIn(("resume", "abc123"), manager.calls)

                with urlopen(base + "/v1/runs/abc123/artifacts/final.mp4") as response:
                    self.assertEqual(response.read(), b"video")

                with self.assertRaises(HTTPError) as missing:
                    urlopen(base + "/v1/runs/missing")
                self.assertEqual(missing.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
