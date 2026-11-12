"""Record-level cleaning of the interim AGMARKNET price table.

Follows the preprocessing list in the proposal: duplicate records, records
with missing or non-positive prices, and records whose prices break the
minimum <= modal <= maximum ordering are removed before any aggregation,
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


def drop_inconsistent_prices(
    df: pd.DataFrame, price_cols: list[str] | None = None
) -> tuple[pd.DataFrame, int]:
    """Drop records that violate min <= modal <= max.

    ``price_cols`` is ordered as (minimum, maximum, modal), matching
    ``config.PRICE_COLUMNS``. Records with a missing price fail the check and
    are dropped too. Returns the kept records and how many were dropped.
    """
    min_col, max_col, modal_col = price_cols or config.PRICE_COLUMNS
    consistent = (df[min_col] <= df[modal_col]) & (df[modal_col] <= df[max_col])
    n_dropped = int((~consistent).sum())
    return df.loc[consistent].reset_index(drop=True), n_dropped


def filter_records(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the record filters in order: duplicates, invalid prices, then
    price consistency."""
    kept, _ = drop_inconsistent_prices(drop_invalid_prices(drop_duplicates(df)))
    return kept
