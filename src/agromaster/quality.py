"""Data quality summary for the converted AGMARKNET price table.

Every function takes the interim price DataFrame (canonical column names from
``config``) and returns a small DataFrame, so the summaries can be shown in a
notebook or written to ``reports/`` without further reshaping.
"""
import pandas as pd

from agromaster import config


def row_count(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row frame with the total record count."""
    return pd.DataFrame({"rows": [len(df)]})


def null_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return nulls and null share per column, in column order."""
    nulls = df.isna().sum()
    share = nulls / len(df) if len(df) else nulls.astype("float64")
    return pd.DataFrame(
        {"column": df.columns, "nulls": nulls.to_numpy(), "null_share": share.to_numpy()}
    )


def date_range(df: pd.DataFrame, date_col: str = config.DATE_COLUMN) -> pd.DataFrame:
    """Return the first and last arrival date, the span in days and distinct dates."""
    dates = df[date_col].dropna()
    first, last = (dates.min(), dates.max()) if not dates.empty else (pd.NaT, pd.NaT)
    span = (last - first).days + 1 if not dates.empty else 0
    return pd.DataFrame(
        {
            "first_date": [first],
            "last_date": [last],
            "span_days": [span],
            "distinct_dates": [dates.nunique()],
        }
    )


def records_per(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return record counts per value of ``column``, largest first.

    Unused categories (present in the dtype but not in the rows) are left out.
    """
    counts = df[column].value_counts(sort=True, dropna=False)
    counts = counts[counts > 0]
    return counts.rename_axis(column).reset_index(name="records")


def records_per_commodity(df: pd.DataFrame) -> pd.DataFrame:
    """Return record counts per commodity, largest first."""
    return records_per(df, "Commodity")


def records_per_state(df: pd.DataFrame) -> pd.DataFrame:
    """Return record counts per state, largest first."""
    return records_per(df, "State")


def quality_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return all quality summaries keyed by name."""
    return {
        "rows": row_count(df),
        "nulls": null_counts(df),
        "date_range": date_range(df),
        "commodity": records_per_commodity(df),
        "state": records_per_state(df),
    }
