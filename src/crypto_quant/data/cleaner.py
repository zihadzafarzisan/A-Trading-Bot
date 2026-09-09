"""Market data cleaning.

Applies deterministic, documented transformations to prepare raw OHLCV data:
deduplication, sorting, gap detection, outlier handling, and type coercion.
The cleaner operates on a copy and never mutates the input; every change is
logged so downstream users know exactly what was altered.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from .validator import DataValidator


@dataclass
class CleaningReport:
    """Summary of cleaning actions applied to a dataset."""

    symbol: str
    timeframe: str
    n_rows_original: int = 0
    n_rows_after: int = 0
    n_duplicates_removed: int = 0
    n_outliers_replaced: int = 0
    n_gaps_detected: int = 0
    actions: List[str] = field(default_factory=list)

    @property
    def n_rows_remaining(self) -> int:
        return self.n_rows_after


class DataCleaner:
    """Cleans OHLCV DataFrames deterministically."""

    OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(
        self,
        outlier_zscore: float = 10.0,
        replace_outliers: bool = True,
        log_actions: bool = True,
    ):
        """Initialize cleaner.

        Args:
            outlier_zscore: A candle is an outlier if its return exceeds this
                many standard deviations. High threshold avoids removing
                legitimate crypto volatility.
            replace_outliers: If True, replace outlier prices with the local
                median (fallback: drop rows).
            log_actions: If True, record each action in the report.
        """
        self.outlier_zscore = outlier_zscore
        self.replace_outliers = replace_outliers
        self.log_actions = log_actions
        self._logger = get_logger("data")

    # ------------------------------------------------------------ main API
    def clean(self, df: pd.DataFrame, symbol: str, timeframe: str) -> tuple[pd.DataFrame, CleaningReport]:
        """Clean a DataFrame and return (cleaned_df, report)."""
        report = CleaningReport(
            symbol=symbol,
            timeframe=timeframe,
            n_rows_original=0 if df is None else len(df),
        )

        if df is None or df.empty:
            report.n_rows_after = 0
            return df, report

        df = df.copy()
        report.n_rows_original = len(df)

        # 1. Column selection & type coercion
        keep = ["timestamp"] + self.OHLCV_COLUMNS
        df = df[keep].copy()
        for col in self.OHLCV_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 2. Drop rows with NaN timestamp
        before = len(df)
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        self._record(report, "Dropped rows with NaN timestamp", before - len(df))

        # 2b. Force consistent float64 dtype for OHLCV columns
        for col in self.OHLCV_COLUMNS:
            df[col] = df[col].astype("float64")

        # 3. Deduplicate on timestamp
        before = len(df)
        df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        report.n_duplicates_removed = before - len(df)
        self._record(report, "Removed duplicate candles", report.n_duplicates_removed)

        # 4. Sort ascending
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 5. Outlier handling on returns
        if len(df) >= 5:
            df = self._handle_outliers(df, report)

        # 6. Gap detection (informational)
        report.n_gaps_detected = self._count_gaps(df, timeframe)
        if report.n_gaps_detected:
            self._record(report, "Detected gaps in timeline", report.n_gaps_detected)

        # 7. Ensure consistent dtypes and reset index
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df = df.reset_index(drop=True)

        report.n_rows_after = len(df)
        return df, report

    # ------------------------------------------------------- internal helpers
    def _record(self, report: CleaningReport, action: str, count: int) -> None:
        """Append an action note to the report if meant to be logged."""
        if self.log_actions and count:
            report.actions.append(f"{action}: {count}")

    def _handle_outliers(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Replace/drop return outliers."""
        returns = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        mean = returns.mean()
        std = returns.std()
        if std == 0 or np.isnan(std):
            return df

        mask = (returns - mean).abs() > self.outlier_zscore * std
        outlier_idx = df.index[mask.fillna(False)]

        if not outlier_idx.any():
            return df

        report.n_outliers_replaced = int(outlier_idx.sum())
        if self.replace_outliers:
            # Replace outlier candles' OHLC with the rolling median of close
            med = df["close"].rolling(5, min_periods=1).median()
            for col in self.OHLCV_COLUMNS:
                df.loc[outlier_idx, col] = med.loc[outlier_idx].values
            self._record(report, "Replaced return-outlier candles with local median",
                         report.n_outliers_replaced)
        else:
            df = df.drop(index=outlier_idx).reset_index(drop=True)
            self._record(report, "Dropped return-outlier candles", report.n_outliers_replaced)
        return df

    def _count_gaps(self, df: pd.DataFrame, timeframe: str) -> int:
        """Count irregular intervals (gaps + outliers)."""
        expected = DataValidator.TIMEFRAME_MS.get(timeframe)
        if expected is None or len(df) < 2:
            return 0
        ts = df["timestamp"].to_numpy()
        diffs = ts[1:] - ts[:-1]
        return int((diffs != expected).sum())
