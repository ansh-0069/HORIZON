from __future__ import annotations

from dataclasses import dataclass, field
import math
import pandas as pd

from src.contracts import CANONICAL_COLUMNS


@dataclass
class QualityReport:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_blocking(self) -> None:
        if self.blockers:
            raise ValueError("Data quality blockers: " + " | ".join(self.blockers))

    def summary(self) -> str:
        return ";".join(self.warnings)


def validate_canonical(frame: pd.DataFrame) -> QualityReport:
    report = QualityReport()
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        report.blockers.append(f"canonical columns missing: {sorted(missing)}")
        return report
    if frame.empty:
        report.blockers.append("no rows after canonicalization")
    for column in ("date", "source_campaign_id"):
        if frame[column].isna().any():
            report.blockers.append(f"null required values in {column}: {int(frame[column].isna().sum())}")
    for column in ("source_system", "source_campaign_id", "channel", "campaign_type", "campaign_name"):
        blanks = frame[column].astype("string").str.strip().fillna("").eq("")
        if blanks.any():
            report.blockers.append(f"blank required values in {column}: {int(blanks.sum())}")
    numeric_required = ("spend", "revenue", "clicks", "impressions", "conversions")
    for column in numeric_required:
        values = pd.to_numeric(frame[column], errors="coerce")
        nulls = values.isna()
        if nulls.any():
            report.blockers.append(f"null or non-numeric required values in {column}: {int(nulls.sum())}")
        non_finite = values.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
        if (~non_finite & ~nulls).any():
            report.blockers.append(f"non-finite required values in {column}: {int((~non_finite & ~nulls).sum())}")
        if (values < 0).any():
            report.blockers.append(f"negative values in {column}")
    # Configured budget is optional, but whenever it is supplied it directly
    # affects delivery uncertainty. Do not let a negative or infinite value
    # alter a forecast merely because missing budgets are otherwise permitted.
    configured_budget = pd.to_numeric(frame["configured_budget"], errors="coerce")
    configured_present = configured_budget.notna()
    configured_non_finite = configured_present & ~configured_budget.map(
        lambda value: math.isfinite(float(value)) if pd.notna(value) else False
    )
    if configured_non_finite.any():
        report.blockers.append(
            f"non-finite configured_budget values: {int(configured_non_finite.sum())}"
        )
    if (configured_budget < 0).any():
        report.blockers.append("negative values in configured_budget")
    keys = ["source_system", "source_campaign_id", "date"]
    conflicts = frame.duplicated(keys, keep=False)
    if conflicts.any():
        report.blockers.append(f"duplicate source campaign-day records: {int(conflicts.sum())}")
    hierarchy = frame.groupby(["source_system", "source_campaign_id"], dropna=False)[["channel", "campaign_type", "campaign_name"]].nunique(dropna=False)
    inconsistent = hierarchy[(hierarchy > 1).any(axis=1)]
    if not inconsistent.empty:
        report.blockers.append(f"inconsistent campaign hierarchy records: {int(len(inconsistent))}")
    null_budget = int(frame["configured_budget"].isna().sum())
    if null_budget:
        report.warnings.append(f"missing configured budget rows={null_budget}")
    quality_flags = frame["quality_flags"].fillna("").astype(str)
    if quality_flags.str.contains(
        "meta_campaign_type_unknown|meta_campaign_type_mapped_unreviewed|"
        "meta_campaign_type_taxonomy_unreviewed_fallback|"
        "meta_campaign_type_mapping_unreviewed_fallback",
        case=False,
    ).any():
        report.warnings.append("campaign taxonomy is incomplete or not explicitly reviewed")
    if quality_flags.str.contains("meta_conversion_treated_as_attributed_revenue", case=False).any():
        report.warnings.append("Meta revenue semantics require documented review")
    if quality_flags.str.contains("source_semantics_unreviewed", case=False).any():
        report.warnings.append("source semantics manifest absent or unreviewed: currency, timezone, and attribution comparability are unreviewed")
    return report
