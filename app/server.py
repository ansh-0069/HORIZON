from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.service import PlannerService


ROOT = Path(__file__).resolve().parents[1]


def make_handler(service: PlannerService):
    class PlannerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "frontend"), **kwargs)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok"})
                return
            if path == "/api/data-health":
                self._json(HTTPStatus.OK, service.data_health())
                return
            if path == "/api/trust":
                self._json(HTTPStatus.OK, service.trust_report())
                return
            if path == "/api/baseline":
                self._json(HTTPStatus.OK, service.forecast({"horizon_days": 60}))
                return
            if path == "/api/evidence-status":
                self._json(HTTPStatus.OK, service.llm_status())
                return
            if path == "/":
                self.path = "/index.html"
            return super().do_GET()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/api/scenario", "/api/optimize", "/api/decisions", "/api/evidence"}:
                self._json(HTTPStatus.NOT_FOUND, {"error": {"code": "NOT_FOUND", "message": "Unknown endpoint"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 100_000:
                    raise ValueError("Request body must be a JSON object smaller than 100 KB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                if path == "/api/optimize":
                    result = service.optimize(payload)
                elif path == "/api/decisions":
                    result = service.record_decision(payload)
                elif path == "/api/evidence":
                    result = service.explain(payload)
                else:
                    result = service.forecast(payload)
                self._json(HTTPStatus.OK, result)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "SCENARIO_INVALID", "message": str(exc)}})
            except Exception as exc:  # pragma: no cover - protected boundary
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"code": "FORECAST_FAILED", "message": str(exc)}})

        def log_message(self, fmt: str, *args) -> None:
            print("planner-api", fmt % args)

    return PlannerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Horizon planner API and UI")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "pickle" / "model.pkl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    service = PlannerService(args.data_dir, args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"Horizon planner available at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
