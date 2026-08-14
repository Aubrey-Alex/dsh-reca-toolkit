from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .jobs import JobManager


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "DSH-ReCA-Gateway/0.1"

    @property
    def manager(self) -> JobManager:
        return self.server.manager  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stdout.write("[reca-gateway] " + (fmt % args) + "\n")

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _body(self) -> dict:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size) if size else b"{}"
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("request body must be a JSON object")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/health":
            return self._send_json(HTTPStatus.OK, {"ok": True, "service": "reca-gateway"})
        if path == "/v1/runs":
            return self._send_json(HTTPStatus.OK, {"runs": self.manager.list_runs()})

        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0:2] == ["v1", "runs"]:
            run_id = parts[2]
            if len(parts) == 3:
                status = self.manager.status(run_id)
                if status is None:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                return self._send_json(HTTPStatus.OK, status)
            if len(parts) == 4 and parts[3] == "events":
                events = self.manager.events(run_id)
                if events is None:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                return self._send_json(HTTPStatus.OK, {"run_id": run_id, "events": events})
            if len(parts) >= 5 and parts[3] == "artifacts":
                relative = "/".join(parts[4:])
                artifact = self.manager.artifact_path(run_id, relative)
                if artifact is None:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "artifact not found"})
                return self._send_file(artifact)
            if len(parts) == 4 and parts[3] == "artifacts":
                status = self.manager.status(run_id)
                if status is None:
                    return self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
                return self._send_json(HTTPStatus.OK, status.get("artifact_manifest", {}))
        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path).rstrip("/")
        try:
            body = self._body()
        except ValueError as exc:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        if path == "/v1/runs":
            try:
                state = self.manager.start(body)
            except (ValueError, TypeError) as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return self._send_json(HTTPStatus.ACCEPTED, state)

        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "cancel":
            state = self.manager.cancel(parts[2])
            if state is None:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
            return self._send_json(HTTPStatus.OK, state)
        if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "resume":
            try:
                state = self.manager.resume(parts[2])
            except (ValueError, TypeError) as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            if state is None:
                return self._send_json(HTTPStatus.NOT_FOUND, {"error": "run not found"})
            return self._send_json(HTTPStatus.ACCEPTED, state)
        return self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})


class GatewayServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], manager: JobManager) -> None:
        super().__init__(address, GatewayHandler)
        self.manager = manager


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose the bundled ReCA pipeline to DSH")
    parser.add_argument("--host", default=os.environ.get("RECA_GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RECA_GATEWAY_PORT", "8787")))
    args = parser.parse_args()
    manager = JobManager()
    server = GatewayServer((args.host, args.port), manager)
    print(f"[reca-gateway] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[reca-gateway] stopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
