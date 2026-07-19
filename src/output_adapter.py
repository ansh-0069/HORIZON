"""Versioned, schema-only boundary for ``predictions.csv``.

This module intentionally owns *all* evaluator-output concerns:

* source-column compatibility aliases;
* target-column names, order, defaults, and primitive types;
* contract validation; and
* atomic CSV writing.

It must not contain forecasting, model, attribution, hierarchy, or business-rule
logic. The incoming DataFrame is an internal forecast representation produced by
``src.forecast``. To support a new evaluator contract, add a new
``OutputSchema`` below (or change ``DEFAULT_SCHEMA_VERSION`` after validation).
No model, pipeline, or runner code should need modification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping

import pandas as pd


ColumnKind = Literal["string", "integer", "number"]
_REQUIRED = object()


class OutputAdapterError(ValueError):
    """Base class for deterministic submission-output contract failures."""


class UnsupportedSchemaVersion(OutputAdapterError):
    """Raised when the caller asks for a schema not registered in this module."""


class SchemaAdaptationError(OutputAdapterError):
    """Raised when the internal forecast cannot be mapped to a target schema."""


class SchemaValidationError(OutputAdapterError):
    """Raised when an adapted output does not satisfy its declared CSV contract."""


@dataclass(frozen=True)
class OutputField:
    """One target CSV column and its compatibility mapping.

    ``sources`` are ordered aliases for the internal forecast field. The adapter
    uses the first alias that exists in an input frame. ``default`` is permitted
    only for explicitly optional presentation fields; required prediction fields
    must not silently receive fabricated values.
    """

    target: str
    sources: tuple[str, ...]
    kind: ColumnKind
    default: Any = _REQUIRED
    allow_null: bool = False

    @property
    def has_default(self) -> bool:
        return self.default is not _REQUIRED


@dataclass(frozen=True)
class OutputSchema:
    """Immutable declaration of one evaluator CSV version.

    Adding a new schema is intentionally data-only: define fields, aliases,
    defaults, and sort keys, then register it in ``SCHEMA_REGISTRY``. The
    evaluator CSV does not receive a synthetic schema-version column unless its
    published contract explicitly requires one.
    """

    version: str
    fields: tuple[OutputField, ...]
    sort_keys: tuple[str, ...] = ()
    require_nonempty: bool = True

    @property
    def columns(self) -> list[str]:
        return [field.target for field in self.fields]

    def assert_well_formed(self) -> None:
        names = self.columns
        if not self.version:
            raise SchemaValidationError("Output schema version must be non-empty")
        if not names or len(names) != len(set(names)):
            raise SchemaValidationError(f"Schema {self.version!r} has missing or duplicate target columns")
        unsupported = [field.kind for field in self.fields if field.kind not in {"string", "integer", "number"}]
        if unsupported:
            raise SchemaValidationError(f"Schema {self.version!r} has unsupported column kinds: {unsupported}")
        unknown_sort_keys = set(self.sort_keys) - set(names)
        if unknown_sort_keys:
            raise SchemaValidationError(f"Schema {self.version!r} sorts by unknown columns: {sorted(unknown_sort_keys)}")


# Current internal forecast -> current submission CSV contract. When organizers
# release the final scorer schema, add a second immutable OutputSchema here,
# retain this one for reproducibility, and switch DEFAULT_SCHEMA_VERSION only
# after contract tests pass.
V1_SCHEMA = OutputSchema(
    version="horizon-v1",
    fields=(
        OutputField("forecast_id", ("forecast_id", "prediction_id"), "string"),
        OutputField("horizon_days", ("horizon_days", "horizon"), "integer"),
        OutputField("level", ("level", "hierarchy_level"), "string"),
        OutputField("channel", ("channel", "source_channel"), "string"),
        OutputField("campaign_type", ("campaign_type", "type"), "string"),
        OutputField("campaign_id", ("campaign_id", "source_campaign_id"), "string", default="ALL"),
        OutputField("campaign_name", ("campaign_name", "source_campaign_name"), "string", default="ALL"),
        OutputField("planned_budget", ("planned_budget", "planned_spend", "budget"), "number"),
        OutputField("predicted_revenue_p10", ("predicted_revenue_p10", "revenue_p10"), "number"),
        OutputField("predicted_revenue_p50", ("predicted_revenue_p50", "revenue_p50", "predicted_revenue"), "number"),
        OutputField("predicted_revenue_p90", ("predicted_revenue_p90", "revenue_p90"), "number"),
        OutputField("predicted_spend_p10", ("predicted_spend_p10", "spend_p10"), "number"),
        OutputField("predicted_spend_p50", ("predicted_spend_p50", "spend_p50", "predicted_spend"), "number"),
        OutputField("predicted_spend_p90", ("predicted_spend_p90", "spend_p90"), "number"),
        OutputField("predicted_roas_p10", ("predicted_roas_p10", "roas_p10"), "number"),
        OutputField("predicted_roas_p50", ("predicted_roas_p50", "roas_p50", "predicted_roas"), "number"),
        OutputField("predicted_roas_p90", ("predicted_roas_p90", "roas_p90"), "number"),
        OutputField("probability_roas_above_target", ("probability_roas_above_target", "roas_target_probability"), "number"),
        OutputField("risk_score", ("risk_score", "forecast_risk_score"), "number"),
        OutputField("quality_flags", ("quality_flags", "flags"), "string", default="none"),
        OutputField("model_version", ("model_version", "artifact_version"), "string"),
    ),
    sort_keys=("horizon_days", "level", "channel", "campaign_type", "campaign_id"),
)

SCHEMA_REGISTRY: Mapping[str, OutputSchema] = {V1_SCHEMA.version: V1_SCHEMA}
DEFAULT_SCHEMA_VERSION = V1_SCHEMA.version

# Forecast calculations remain upstream, but the public CSV needs a stable
# text representation. Six digits after the decimal are far more precise
# than the monetary/ROAS decision inputs while eliminating harmless
# sub-nanounit renderer differences across supported Python/NumPy builds.
# This is serialization policy only, not rounding used by the model.
CSV_FLOAT_FORMAT = "%.6f"

# Transitional public constant for existing callers. It is defined here rather
# than in contracts.py so this module remains the single owner of CSV schema.
FORECAST_COLUMNS = V1_SCHEMA.columns


def get_schema(version: str | None = None, registry: Mapping[str, OutputSchema] = SCHEMA_REGISTRY) -> OutputSchema:
    """Resolve and validate a registered schema declaration."""
    requested = version or DEFAULT_SCHEMA_VERSION
    try:
        schema = registry[requested]
    except KeyError as exc:
        raise UnsupportedSchemaVersion(
            f"Unsupported predictions.csv schema version {requested!r}; supported versions: {sorted(registry)}"
        ) from exc
    schema.assert_well_formed()
    return schema


class OutputAdapter:
    """Adapter-pattern implementation for internal forecast -> evaluator CSV.

    The adapter is deliberately stateless. It can receive a test-local registry,
    which makes future schema compatibility tests independent of global process
    state and avoids runtime schema mutation.
    """

    def __init__(
        self,
        registry: Mapping[str, OutputSchema] = SCHEMA_REGISTRY,
        default_schema_version: str = DEFAULT_SCHEMA_VERSION,
    ) -> None:
        self._registry = dict(registry)
        self._default_schema_version = default_schema_version

    def schema(self, version: str | None = None) -> OutputSchema:
        return get_schema(version or self._default_schema_version, self._registry)

    @staticmethod
    def _source_series(frame: pd.DataFrame, field: OutputField) -> pd.Series:
        source = next((candidate for candidate in field.sources if candidate in frame.columns), None)
        if source is not None:
            series = frame[source].copy()
        elif field.has_default:
            series = pd.Series(field.default, index=frame.index)
        else:
            raise SchemaAdaptationError(
                f"Cannot populate output column {field.target!r}; expected one of source columns {list(field.sources)}"
            )
        if field.has_default:
            series = series.fillna(field.default)
            if field.kind == "string":
                blank = series.astype("string").str.strip().eq("")
                series = series.mask(blank, field.default)
        return series

    @staticmethod
    def _coerce(series: pd.Series, field: OutputField) -> pd.Series:
        if field.kind == "string":
            if not field.allow_null and series.isna().any():
                raise SchemaValidationError(f"Output column {field.target!r} contains null values")
            return series.astype("string") if field.allow_null else series.astype(str)

        numeric = pd.to_numeric(series, errors="coerce")
        if not field.allow_null and numeric.isna().any():
            raise SchemaValidationError(f"Output column {field.target!r} contains non-numeric or null values")
        finite = numeric.dropna().map(lambda value: math.isfinite(float(value)))
        if not finite.all():
            raise SchemaValidationError(f"Output column {field.target!r} contains non-finite numeric values")
        if field.kind == "integer":
            non_integral = numeric.dropna().map(lambda value: float(value).is_integer())
            if not non_integral.all():
                raise SchemaValidationError(f"Output column {field.target!r} requires integer values")
            return numeric.astype("Int64") if field.allow_null else numeric.astype("int64")
        return numeric.astype("float64")

    def adapt(self, frame: pd.DataFrame, schema_version: str | None = None) -> pd.DataFrame:
        """Map one internal forecast frame to a versioned, ordered CSV frame."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("OutputAdapter.adapt requires a pandas DataFrame")
        schema = self.schema(schema_version)
        output = pd.DataFrame(index=frame.index)
        for field in schema.fields:
            output[field.target] = self._coerce(self._source_series(frame, field), field)
        if schema.sort_keys:
            output = output.sort_values(list(schema.sort_keys), kind="stable")
        output = output.reset_index(drop=True)
        self.validate(output, schema.version)
        return output

    def validate(self, frame: pd.DataFrame, schema_version: str | None = None) -> None:
        """Validate CSV shape, primitive types, nullability, and column order only.

        Forecast correctness (for example, quantile ordering or ROAS semantics)
        belongs to the forecasting domain, not this serialization boundary.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("OutputAdapter.validate requires a pandas DataFrame")
        schema = self.schema(schema_version)
        if list(frame.columns) != schema.columns:
            raise SchemaValidationError(
                f"Schema {schema.version!r} requires columns {schema.columns}; received {list(frame.columns)}"
            )
        if schema.require_nonempty and frame.empty:
            raise SchemaValidationError(f"Schema {schema.version!r} does not permit an empty predictions.csv")
        for field in schema.fields:
            self._coerce(frame[field.target], field)

    def write(self, frame: pd.DataFrame, output_path: Path, schema_version: str | None = None) -> pd.DataFrame:
        """Adapt, validate, and atomically replace ``predictions.csv``.

        The returned DataFrame is exactly what was serialized. A failure leaves a
        pre-existing destination untouched and removes any temporary file.
        """
        output = self.adapt(frame, schema_version)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=".csv",
                prefix=".predictions-",
                dir=output_path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                output.to_csv(
                    handle,
                    index=False,
                    float_format=CSV_FLOAT_FORMAT,
                    lineterminator="\n",
                )
            temporary_path.replace(output_path)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return output


def to_submission_schema(frame: pd.DataFrame, schema_version: str | None = None) -> pd.DataFrame:
    """Compatibility facade for callers that need an adapted DataFrame only."""
    return OutputAdapter().adapt(frame, schema_version)


def validate_submission_schema(frame: pd.DataFrame, schema_version: str | None = None) -> None:
    """Compatibility facade for schema-only validation."""
    OutputAdapter().validate(frame, schema_version)


def write_predictions_csv(frame: pd.DataFrame, output_path: Path, schema_version: str | None = None) -> pd.DataFrame:
    """The sole public writer for evaluator ``predictions.csv`` files."""
    return OutputAdapter().write(frame, output_path, schema_version)
