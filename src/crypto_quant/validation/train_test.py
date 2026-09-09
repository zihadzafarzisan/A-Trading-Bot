"""Train/test validation.

Strictly chronological time-series splitting — never random. Guards against
overlap and out-of-order dates. Supports the default research structure
(2020-2024 train / 2025 validation / 2026 out-of-sample) with configurable
boundaries.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from ..logging_config import get_logger

logger = get_logger("validation")


@dataclass
class SplitDates:
    """Chronological split boundaries (inclusive end dates)."""

    train_end: str          # e.g. "2024-12-31"
    validation_end: str     # e.g. "2025-12-31"
    test_end: str           # e.g. "2026-09-08"

    def validate(self) -> None:
        """Ensure strictly increasing boundaries."""
        if not (self.train_end < self.validation_end < self.test_end):
            raise ValueError(
                f"Split boundaries must be strictly increasing: "
                f"{self.train_end} < {self.validation_end} < {self.test_end}"
            )


class TrainTestSplitter:
    """Splits a time-series DataFrame into train/validation/test."""

    DEFAULT_DATES = SplitDates(
        train_end="2024-12-31",
        validation_end="2025-12-31",
        test_end="2026-09-08",
    )

    def __init__(self, dates: Optional[SplitDates] = None):
        self.dates = dates or self.DEFAULT_DATES
        self.dates.validate()

    # ------------------------------------------------------------ main API
    def split(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into (train, validation, test).

        Args:
            df: OHLCV/feature DataFrame with a timestamp column (unix ms).
            timestamp_col: Column holding timestamps.

        Returns:
            (train, validation, test) as contiguous, non-overlapping slices
            sorted ascending.
        """
        self._assert_chronological(df, timestamp_col)
        t = pd.to_datetime(df[timestamp_col], unit="ms", utc=True)

        train_end_dt = pd.Timestamp(self.dates.train_end, tz="UTC")
        val_end_dt = pd.Timestamp(self.dates.validation_end, tz="UTC")
        test_end_dt = pd.Timestamp(self.dates.test_end, tz="UTC")

        train_mask = t <= train_end_dt
        val_mask = (t > train_end_dt) & (t <= val_end_dt)
        test_mask = (t > val_end_dt) & (t <= test_end_dt)

        train = df[train_mask].copy()
        validation = df[val_mask].copy()
        test = df[test_mask].copy()

        if len(train) == 0 or len(test) == 0:
            logger.warning(
                "Split produced empty slices (train=%d val=%d test=%d) — check data coverage",
                len(train), len(validation), len(test),
            )
        return train, validation, test

    # ------------------------------------------------------------ helpers
    def _assert_chronological(self, df: pd.DataFrame, timestamp_col: str) -> None:
        if timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column: {timestamp_col}")
        t = df[timestamp_col].to_numpy()
        if len(t) > 1 and not (t[1:] >= t[:-1]).all():
            raise ValueError("Data must be sorted ascending by timestamp (no look-ahead splits)")

    def split_by_ratio(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp",
        train_ratio: float = 0.7, val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological split by fraction (train/val/test). Never random."""
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError("Ratios must satisfy 0 < train < 1, 0 < val < 1, train+val < 1")
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train = df.iloc[:train_end].copy()
        validation = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        return train, validation, test

    def describe(self) -> Dict[str, str]:
        """Return the split configuration."""
        return {
            "train_end": self.dates.train_end,
            "validation_end": self.dates.validation_end,
            "test_end": self.dates.test_end,
        }