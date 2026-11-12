"""Record-level cleaning of the interim AGMARKNET price table.

Follows the preprocessing list in the proposal: duplicate records and records
with missing or non-positive prices are removed before any aggregation,
alignment or feature engineering. Each step is a small function that takes
and returns a pandas DataFrame and leaves its input unchanged.
"""
import pandas as pd

from agromaster import config


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate records, keeping the first occurrence."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def drop_invalid_prices(
    df: pd.DataFrame, price_cols: list[str] | None = None
) -> pd.DataFrame:
    """Drop records where any price is missing, zero or negative."""
    cols = price_cols or config.PRICE_COLUMNS
    valid = df[cols].notna().all(axis=1) & (df[cols] > 0).all(axis=1)
    return df.loc[valid].reset_index(drop=True)


def filter_records(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the record filters in order: duplicates, then invalid prices."""
    return drop_invalid_prices(drop_duplicates(df))
