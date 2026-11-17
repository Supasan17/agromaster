"""Chronological data splitting without target leakage.

Follows the proposal's "Data splitting and validation" subsection: records
are split by date, never shuffled, so every record sharing a date lands in
the same partition. A training-side record on date ``t`` has its label built
from the price on ``t + HORIZON_DAYS``, so training records whose target date
reaches the later partition are purged. Each step takes and returns pandas
DataFrames and leaves its input unchanged; rows keep their original order and
index.
"""
import pandas as pd

from agromaster import config


def _dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Return the row dates of ``df`` normalised to midnight."""
    return pd.to_datetime(df[date_col]).dt.normalize()


def purge_target_overlap(
    df: pd.DataFrame,
    boundary: str | pd.Timestamp,
    date_col: str = config.DATE_COLUMN,
    horizon_days: int = config.HORIZON_DAYS,
) -> pd.DataFrame:
    """Drop rows whose target date falls on or after ``boundary``.

    The target date of a row on date ``t`` is ``t + horizon_days`` calendar
    days. Rows are kept only when that date is strictly before
    ``boundary`` (the first day of the later partition), so no kept row's
    label uses a price from that partition. Rows with a missing date are
    dropped.
    """
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be non-negative, got {horizon_days}")
    start = pd.Timestamp(boundary).normalize()
    target = _dates(df, date_col) + pd.Timedelta(days=horizon_days)
    return df[(target < start).to_numpy()].copy()


def chronological_split(
    df: pd.DataFrame,
    dev_end: str | pd.Timestamp = config.DEV_END,
    test_start: str | pd.Timestamp = config.TEST_START,
    date_col: str = config.DATE_COLUMN,
    horizon_days: int = config.HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into a development frame and a final test frame.

    Development rows are dated on or before ``dev_end`` and test rows on or
    after ``test_start``; rows dated between the two, or without a date,
    are in neither. Splitting on whole dates keeps all records of a date
    together. Development rows whose ``horizon_days``-ahead target date
    reaches ``test_start`` are purged with :func:`purge_target_overlap`.
    Returns ``(dev, test)``.
    """
    end = pd.Timestamp(dev_end).normalize()
    start = pd.Timestamp(test_start).normalize()
    if end >= start:
        raise ValueError(
            f"dev_end ({end.date()}) must be before test_start ({start.date()})"
        )
    dates = _dates(df, date_col)
    dev = df[(dates <= end).to_numpy()]
    test = df[(dates >= start).to_numpy()].copy()
    dev = purge_target_overlap(dev, start, date_col, horizon_days)
    return dev, test
