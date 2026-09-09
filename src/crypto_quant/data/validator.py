"""Market data validation.

Detects and reports data-quality problems: duplicate candles, missing candles,
gaps, out-of-order timestamps, corrupted/negative values, and non-monotonic
high/low structure. Validators never silently discard data — they report, so
callers decide how to handle.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ValidationIssue:
    """A single data-quality issue."""

    timestamp: Optional[int]
    issue_type: str  # duplicate, gap, out_of_order, invalid_value, etc.
    severity: str    # info, warning, error
    message: str


@dataclass
class ValidationReport:
    """Aggregate validation results for a dataset."""

    symbol: str
    timeframe: str
    n_candles: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_clean(self) -> bool:
        """True if there are no error-severity issues."""
        return self.error_count == 0

    def summary(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_candles": self.n_candles,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "clean": self.is_clean,
        }


class DataValidator:
    """Validates OHLCV DataFrames for common data-quality problems."""

    # Columns required for validation
    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    # Timeframe -> expected ms spacing
    TIMEFRAME_MS = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }

    def __init__(self, strict: bool = True):
        """Initialize validator.

        Args:
            strict: If True, invalid OHLCV structure counts as an error;
                otherwise it is only a warning.
        """
        self.strict = strict

    # ------------------------------------------------------------ main API
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        sort: bool = True,
    ) -> ValidationReport:
        """Validate a DataFrame and return an issue report."""
        report = ValidationReport(symbol=symbol, timeframe=timeframe, n_candles=len(df))

        if df is None or df.empty:
            report.issues.append(ValidationIssue(None, "empty_dataset", "error", "Dataset is empty"))
            return report

        # Missing required columns
        missing = [c for c in self.OHLCV_COLUMNS if c not in df.columns]
        if missing:
            report.issues.append(
                ValidationIssue(None, "missing_columns", "error", f"Missing columns: {missing}")
            )
            return report

        df = df.copy()
        # Sort by timestamp ascending
        if sort:
            before_monotonic = df["timestamp"].is_monotonic_increasing
            if not before_monotonic:
                df = df.sort_values("timestamp").reset_index(drop=True)
                report.issues.append(
                    ValidationIssue(None, "out_of_order", "warning",
                                    "Data was out of time order; sorted ascending")
                )

        self._validate_duplicates(df, report)
        self._validate_spacing(df, timeframe, report)
        self._validate_values(df, report)

        return report

    # ------------------------------------------------------- internal checks
    def _validate_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Detect duplicate timestamps."""
        dupes = df[df["timestamp"].duplicated(keep=False)]
        if not dupes.empty:
            n = dupes["timestamp"].nunique()
            report.issues.append(
                ValidationIssue(
                    int(dupes["timestamp"].iloc[0]), "duplicate",
                    "warning", f"{len(dupes)} duplicate candles across {n} timestamps",
                )
            )

    def _validate_spacing(self, df: pd.DataFrame, timeframe: str, report: ValidationReport) -> None:
        """Detect gaps and missing candles using expected spacing."""
        expected = self.TIMEFRAME_MS.get(timeframe)
        if expected is None:
            report.issues.append(
                ValidationIssue(None, "unknown_timeframe", "warning",
                                f"No expected spacing defined for '{timeframe}'")
            )
            return

        ts = df["timestamp"].to_numpy()
        if len(ts) < 2:
            return

        diffs = ts[1:] - ts[:-1]
        bad = diffs != expected

        if bad.any():
            n_gaps = int(bad.sum())
            # Estimate missing candles from the total delta
            span = ts[-1] - ts[0]
            expected_count = span // expected + 1
            missing = int(max(0, expected_count - len(df)))
            report.issues.append(
                ValidationIssue(
                    int(ts[0]), "gap", "warning",
                    f"{n_gaps} irregular interval(s); ~{missing} missing candle(s) expected",
                )
            )

    def _validate_values(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Validate OHLCV value integrity (negatives, high<low, etc.)."""
        for col in ("open", "high", "low", "close", "volume"):
            neg = df[col] < 0
            if neg.any():
                report.issues.append(
                    ValidationIssue(int(df.loc[neg, "timestamp"].iloc[0]), "invalid_value",
                                    "error" if self.strict else "warning",
                                    f"Negative value in {col} at {int(neg.sum())} candle(s)")
                )

        # high < low
        invalid_hilo = df["high"] < df["low"]
        if invalid_hilo.any():
            report.issues.append(
                ValidationIssue(int(df.loc[invalid_hilo, "timestamp"].iloc[0]), "invalid_value",
                                "error" if self.strict else "warning",
                                f"high < low at {int(invalid_hilo.sum())} candle(s)")
            )

        # high < open or high < close
        for col in ("open", "close"):
            m = df["high"] < df[col]
            if m.any():
                report.issues.append(
                    ValidationIssue(int(df.loc[m, "timestamp"].iloc[0]), "invalid_value",
                                    "error" if self.strict else "warning",
                                    f"high < {col} at {int(m.sum())} candle(s)")
                )

        # NaN in OHLCV
        nan_mask = df[["open", "high", "low", "close", "volume"]].isna()
        if nan_mask.any().any():
            report.issues.append(
                ValidationIssue(None, "nan_values", "warning",
                                f"NaN values present in {int(nan_mask.sum().sum())} cell(s)")
            )

        # Zero/negative volume
        bad_vol = df["volume"] <= 0
        if bad_vol.any():
            report.issues.append(
                ValidationIssue(int(df.loc[bad_vol, "timestamp"].iloc[0]), "invalid_value",
                                "warning", f"Non-positive volume at {int(bad_vol.sum())} candle(s)")
            )


class DataQualityReport:
    """Aggregate validation results across many datasets (for summary views)."""

    def __init__(self) -> None:
        self.reports: List[ValidationReport] = []

    def add(self, report: ValidationReport) -> None:
        self.reports.append(report)

    def total_issues(self) -> int:
        return sum(len(r.issues) for r in self.reports)

    def total_errors(self) -> int:
        return sum(r.error_count for r in self.reports)

    def total_warnings(self) -> int:
        return sum(r.warning_count for r in self.reports)

    def datasets_clean(self) -> int:
        return sum(1 for r in self.reports if r.is_clean)

    def datasets_total(self) -> int:
        return len(self.reports)
