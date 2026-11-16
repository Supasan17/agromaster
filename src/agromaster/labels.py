"""Target construction for the price-decline risk label.

The label for a prediction date ``t`` starts from the ratio

    r_t = modal(t + HORIZON_DAYS) / mean modal over the HISTORY_DAYS days before t

as defined in the proposal's target subsection. Both prices come from the
observed modal price (never the forward-filled predictor copy), so a day
without a report gives no future price and the target is never filled.
Future prices are used here only to build labels and are never returned as
predictors. Each step takes and returns a pandas DataFrame and leaves its
input unchanged.
"""
import numpy as np
import pandas as pd

from agromaster import config
from agromaster.features import _lagged_values, _series_ids

MODAL_COLUMN = "Modal_Price"


def _history_mean(
    df: pd.DataFrame,
    days: int,
    min_periods: int,
    keys: list[str] | None,
    date_col: str,
    price_col: str,
) -> np.ndarray:
    """Return the mean of ``price_col`` over the ``days`` days before each date.

    The window for a row on date ``t`` covers ``t - days`` to ``t - 1``
    inclusive in the same series, so it excludes ``t`` and every later day.
    Missing prices are skipped; the result is missing when fewer than
    ``min_periods`` prices are observed. Rows on the same series and date
    share a window.
    """
    if df.empty:
        return np.empty(0, dtype="float32")
    work = pd.DataFrame(
        {
            "_series": _series_ids(df, keys),
            "_date": pd.to_datetime(df[date_col]).dt.normalize().to_numpy(),
            "_price": df[price_col].to_numpy(dtype="float64"),
            "_row": np.arange(len(df)),
        }
    )
    work = work.dropna(subset=["_date"]).sort_values(
        ["_series", "_date", "_row"], kind="stable"
    )
    values = (
        work.set_index("_date")
        .groupby("_series", sort=False)["_price"]
        .rolling(f"{days}D", min_periods=min_periods, closed="left")
        .mean()
        .to_numpy()
    )
    keyed = pd.Series(values, index=pd.MultiIndex.from_frame(work[["_series", "_date"]]))
    values = keyed.groupby(level=[0, 1], sort=False).transform("first").to_numpy()
    result = np.full(len(df), np.nan, dtype="float32")
    result[work["_row"].to_numpy()] = values
    return result


def add_future_ratio(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    date_col: str = config.DATE_COLUMN,
    price_col: str = MODAL_COLUMN,
    horizon_days: int = config.HORIZON_DAYS,
    history_days: int = config.HISTORY_DAYS,
) -> pd.DataFrame:
    """Add the ``horizon_days``-ahead price ratio used to define the label.

    Within each series (``keys`` present in ``df``, default
    ``config.SERIES_KEYS``):

    * ``config.FUTURE_PRICE_COLUMN`` is ``price_col`` exactly
      ``horizon_days`` calendar days after the row's date (missing when that
      day has no observed price);
    * ``config.HISTORY_MEAN_COLUMN`` is the mean of ``price_col`` over the
      ``history_days`` calendar days before the row's date, excluding the
      date itself (at least ``config.HISTORY_MIN_PERIODS`` observed prices);
    * ``config.RATIO_COLUMN`` is the future price divided by the history
      mean, missing when either is missing or the mean is not positive.

    ``price_col`` must be the observed (unfilled) modal price. Dates are
    matched on the calendar, not row positions, so the result does not
    depend on row order, and rows keep their original order and index. The
    future price and ratio are label inputs only and must not be used as
    predictors.
    """
    out = df.copy()
    future = _lagged_values(df, -horizon_days, keys, date_col, price_col)
    history = _history_mean(
        df, history_days, config.HISTORY_MIN_PERIODS, keys, date_col, price_col
    )
    safe_history = np.where(history > 0, history, np.nan)
    out[config.FUTURE_PRICE_COLUMN] = future
    out[config.HISTORY_MEAN_COLUMN] = history
    out[config.RATIO_COLUMN] = (future / safe_history).astype("float32")
    return out
