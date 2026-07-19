from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from product.app.service import PlannerService


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PRODUCT_ROOT.parent
LOGGER = logging.getLogger("horizon.planner")
LOCAL_ONLY_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def validate_live_llm_host(host: str, enabled: bool) -> None:
    """Prevent the optional network narrator from being exposed beyond localhost."""
    if enabled and host.strip().lower() not in LOCAL_ONLY_HOSTS:
        raise ValueError(
            "--enable-live-llm is allowed only with --host 127.0.0.1, ::1, or localhost"
        )


def _reject_nonstandard_json_constant(value: str) -> None:
    # Python's json parser accepts NaN/Infinity by default. The service rejects
    # non-finite values too, but reject them at the HTTP boundary for a clearer
    # contract and to avoid accidentally accepting non-standard JSON.
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def make_handler(service: PlannerService):
    class PlannerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PRODUCT_ROOT / "frontend"), **kwargs)

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
                payload = json.loads(
                    self.rfile.read(length).decode("utf-8"),
                    parse_constant=_reject_nonstandard_json_constant,
                )
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
            except Exception:  # pragma: no cover - protected boundary
                # Preserve diagnostics for the local operator without exposing
                # filesystem paths, model internals, or provider details to a
                # browser/API client.
                LOGGER.exception("planner_api_failure endpoint=%s", path)
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": {
                            "code": "FORECAST_FAILED",
                            "message": "The planner could not complete this request. Review the local server log for details.",
                        }
                    },
                )

        def log_message(self, fmt: str, *args) -> None:
            # Standard request logging avoids ad-hoc prints while retaining a
            # useful local-demo audit trail. Request bodies are never logged.
            LOGGER.info("planner_api_request %s", fmt % args)

    return PlannerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Horizon planner API and UI")
    parser.add_argument("--data-dir", type=Path, default=PRODUCT_ROOT / "demo_data")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "pickle" / "model.pkl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument(
        "--enable-live-llm",
        action="store_true",
        help="Allow the optional live narrator for explicit API requests on localhost only.",
    )
    args = parser.parse_args()
    try:
        validate_live_llm_host(args.host, args.enable_live_llm)
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    service = PlannerService(args.data_dir, args.model, allow_live_llm=args.enable_live_llm)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    LOGGER.info(
        "planner_server_started host=%s port=%s live_llm_enabled=%s",
        args.host,
        args.port,
        args.enable_live_llm,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
