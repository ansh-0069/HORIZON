from __future__ import annotations

from dataclasses import dataclass, field
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
    for column in ("date", "spend", "revenue", "source_campaign_id"):
        if frame[column].isna().any():
            report.blockers.append(f"null required values in {column}: {int(frame[column].isna().sum())}")
    for column in ("spend", "revenue"):
        if (frame[column] < 0).any():
            report.blockers.append(f"negative values in {column}")
    keys = ["source_system", "source_campaign_id", "date"]
    conflicts = frame.duplicated(keys, keep=False)
    if conflicts.any():
        report.blockers.append(f"duplicate source campaign-day records: {int(conflicts.sum())}")
    null_budget = int(frame["configured_budget"].isna().sum())
    if null_budget:
        report.warnings.append(f"missing configured budget rows={null_budget}")
    if frame["quality_flags"].str.contains("unknown|requires_semantic", case=False, na=False).any():
        report.warnings.append("taxonomy or Meta revenue semantic requires documented review")
    return report
