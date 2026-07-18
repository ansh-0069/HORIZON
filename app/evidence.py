"""Grounded, optional LLM narrative generation for Horizon.

This module is deliberately outside the submission runner.  It never receives raw
marketing records and it cannot change a forecast or an optimization result.  Its
only input is a compact, deterministic evidence packet produced after the model
has finished predicting.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-luna"
_NUMERIC_CLAIM = re.compile(r"\d")
_CAUSAL_LANGUAGE = re.compile(r"\b(caus\w*|driv\w*|lift|incremental|guarante\w*|proven?)\b", re.IGNORECASE)


class EvidenceGenerationError(RuntimeError):
    """A safe, user-displayable error raised when narrative generation is unavailable."""


@dataclass(frozen=True)
class EvidenceClientConfig:
    api_key: str
    model: str
    timeout_seconds: float = 20.0


def _read_local_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without logging or mutating process environment."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_evidence_config(root: Path = ROOT) -> EvidenceClientConfig | None:
    """Find an API key in the environment or local development env file.

    The secret is returned only to the HTTP client and is never included in a
    response, exception, log message, or evidence packet.
    """
    local = _read_local_env(root / ".env.local")
    api_key = os.environ.get("OPENAI_API_KEY") or local.get("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.environ.get("HORIZON_LLM_MODEL") or local.get("HORIZON_LLM_MODEL") or DEFAULT_MODEL
    return EvidenceClientConfig(api_key=api_key, model=model)


def evidence_status(root: Path = ROOT) -> dict[str, Any]:
    config = load_evidence_config(root)
    return {
        "configured": config is not None,
        "model": config.model if config else None,
        "mode": "optional_grounded_narrative",
        "prediction_dependency": False,
    }


def build_evidence_packet(evidence: Mapping[str, Any], overall: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the only model-visible source of truth for a narrative.

    Numeric values remain here as facts, but the output contract prohibits new
    numerical claims. This makes source citation and validation tractable in the
    MVP, while the UI continues to render model numbers from the deterministic
    forecast response rather than from LLM prose.
    """
    target = float(evidence["target_roas"])
    signals: list[dict[str, Any]] = [
        {
            "id": "forecast_boundary",
            "kind": "method_limit",
            "statement": "The model provides a conditional attribution-based forecast, not a causal incrementality estimate.",
        },
        {
            "id": "overall_guardrail",
            "kind": "forecast",
            "statement": "The modeled ROAS guardrail probability and risk score determine the deterministic decision posture.",
            "metrics": {
                "target_roas": target,
                "probability_above_target": float(overall["probability_roas_above_target"]),
                "risk_score": float(overall["risk_score"]),
            },
        },
        {
            "id": "forecast_range",
            "kind": "forecast",
            "statement": "Revenue uncertainty is represented by an empirical P10 to P90 interval, not a point promise.",
            "metrics": {
                "revenue_p10": float(overall["predicted_revenue_p10"]),
                "revenue_p50": float(overall["predicted_revenue_p50"]),
                "revenue_p90": float(overall["predicted_revenue_p90"]),
            },
        },
    ]
    for driver in evidence.get("drivers", []):
        channel = str(driver["channel"])
        signals.append(
            {
                "id": f"channel:{channel}",
                "kind": "channel_forecast",
                "statement": "This channel appears among the highest modeled revenue contributors in this scenario.",
                "metrics": {
                    "channel": channel,
                    "expected_revenue": float(driver["expected_revenue"]),
                    "expected_roas": float(driver["expected_roas"]),
                },
            }
        )
    for index, risk in enumerate(evidence.get("risks", []), start=1):
        signals.append({"id": f"risk:{index}", "kind": "risk", "statement": str(risk)})
    return {
        "forecast_id": str(overall["forecast_id"]),
        "deterministic_decision": str(evidence["decision"]),
        "causal_status": "observational_association",
        "signals": signals,
    }


BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": ["approve", "revise_or_test"]},
        "causal_status": {"type": "string", "enum": ["observational_association"]},
        "headline": {"type": "string", "minLength": 1, "maxLength": 220},
        "facts": {"type": "array", "items": {"$ref": "#/$defs/cited_item"}, "maxItems": 3},
        "assumptions": {"type": "array", "items": {"$ref": "#/$defs/cited_item"}, "maxItems": 3},
        "recommendations": {"type": "array", "items": {"$ref": "#/$defs/cited_item"}, "maxItems": 3},
        "limitations": {"type": "array", "items": {"$ref": "#/$defs/cited_item"}, "maxItems": 3},
    },
    "required": ["decision", "causal_status", "headline", "facts", "assumptions", "recommendations", "limitations"],
    "$defs": {
        "cited_item": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 320},
                "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            },
            "required": ["text", "evidence_ids"],
        }
    },
}


SYSTEM_PROMPT = """You are Horizon's evidence narrator for a paid-media planner.
You do not forecast, optimize, calculate, or change the deterministic decision.
Use only the supplied evidence packet. Cite one or more exact evidence IDs for
every list item. Do not make numerical claims, repeat numeric values, infer
causality, claim lift, guarantee an outcome, or present observational patterns as
causal effects. If evidence is insufficient, state that limitation and recommend
a bounded validation experiment. Keep the language concise and decision-useful."""


def _extract_output_text(payload: Mapping[str, Any]) -> str:
    for item in payload.get("output", []):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise EvidenceGenerationError("The model returned no structured text output")


def _validate_text(text: Any, valid_ids: set[str]) -> None:
    if not isinstance(text, str) or not text.strip() or len(text) > 320:
        raise EvidenceGenerationError("The model returned an invalid narrative item")
    if _NUMERIC_CLAIM.search(text) or _CAUSAL_LANGUAGE.search(text):
        raise EvidenceGenerationError("The model narrative exceeded its evidence boundary")


def validate_brief(candidate: Any, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Defence in depth beyond provider-side structured-output enforcement."""
    if not isinstance(candidate, Mapping):
        raise EvidenceGenerationError("The model response was not an object")
    required = {"decision", "causal_status", "headline", "facts", "assumptions", "recommendations", "limitations"}
    if set(candidate) != required:
        raise EvidenceGenerationError("The model response did not match the approved brief contract")
    if candidate["decision"] != packet["deterministic_decision"]:
        raise EvidenceGenerationError("The model attempted to change the deterministic decision")
    if candidate["causal_status"] != "observational_association":
        raise EvidenceGenerationError("The model attempted to change the causal-status boundary")
    _validate_text(candidate["headline"], {"headline"})
    valid_ids = {str(signal["id"]) for signal in packet["signals"]}
    result: dict[str, Any] = {
        "decision": candidate["decision"],
        "causal_status": candidate["causal_status"],
        "headline": candidate["headline"].strip(),
    }
    for name in ("facts", "assumptions", "recommendations", "limitations"):
        items = candidate[name]
        if not isinstance(items, list) or len(items) > 3:
            raise EvidenceGenerationError("The model returned an invalid narrative collection")
        clean_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping) or set(item) != {"text", "evidence_ids"}:
                raise EvidenceGenerationError("The model returned an uncitable narrative item")
            _validate_text(item.get("text"), valid_ids)
            ids = item.get("evidence_ids")
            if not isinstance(ids, list) or not ids or any(not isinstance(identifier, str) or identifier not in valid_ids for identifier in ids):
                raise EvidenceGenerationError("The model cited evidence outside the approved packet")
            clean_items.append({"text": item["text"].strip(), "evidence_ids": list(dict.fromkeys(ids))})
        result[name] = clean_items
    return result


class OpenAIEvidenceClient:
    """Small standard-library Responses API client used only by the local planner."""

    def __init__(self, config: EvidenceClientConfig) -> None:
        self.config = config

    def generate(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(packet, sort_keys=True, separators=(",", ":"))},
            ],
            "text": {"format": {"type": "json_schema", "name": "horizon_evidence_brief", "strict": True, "schema": BRIEF_SCHEMA}},
            "store": False,
        }
        request = Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # API errors are safe to classify, but deliberately never surface the
            # raw response body because provider messages can change over time.
            error_code = ""
            try:
                error_payload = json.loads(exc.read().decode("utf-8"))
                error_code = str(error_payload.get("error", {}).get("code", ""))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                pass
            if exc.code == 429 and error_code == "insufficient_quota":
                raise EvidenceGenerationError("OpenAI API credits or project quota are unavailable; showing deterministic evidence instead") from exc
            if exc.code == 429:
                raise EvidenceGenerationError("OpenAI narrative service is rate-limited; retry shortly or use deterministic evidence") from exc
            if exc.code in {401, 403}:
                raise EvidenceGenerationError("OpenAI narrative service rejected the configured project access") from exc
            raise EvidenceGenerationError(f"OpenAI narrative service failed with HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise EvidenceGenerationError("OpenAI narrative service is temporarily unavailable") from exc
        try:
            return validate_brief(json.loads(_extract_output_text(raw_response)), packet)
        except json.JSONDecodeError as exc:
            raise EvidenceGenerationError("The model did not return valid structured JSON") from exc
