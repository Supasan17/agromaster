"""Feature engineering for the price-decline risk classifier.

Builds the derived predictors listed in the proposal's feature table from the
calendar-aligned series produced by ``agromaster.cleaning``. Predictors use
only information available on the prediction date, so they are computed from
the forward-filled predictor copies of the prices (``<col>`` +
``config.FILLED_SUFFIX``) when those are present, and never from future
target prices. Each step is a small function that takes and returns a pandas
DataFrame and leaves its input unchanged.
"""
import numpy as np
import pandas as pd

from agromaster import config


def predictor_price_columns(df: pd.DataFrame) -> list[str]:
    """Return the (minimum, maximum, modal) price columns to use as predictors.

    The forward-filled copies are used when ``df`` has all of them (aligned
    data); otherwise the observed price columns are used.
    """
    filled = [c + config.FILLED_SUFFIX for c in config.PRICE_COLUMNS]
    if all(c in df.columns for c in filled):
        return filled
    return list(config.PRICE_COLUMNS)


def add_price_spread(
    df: pd.DataFrame, price_cols: list[str] | None = None
) -> pd.DataFrame:
    """Add the price spread and spread ratio (features 6 and 7).

    ``config.SPREAD_COLUMN`` is maximum price minus minimum price and
    ``config.SPREAD_RATIO_COLUMN`` is that spread divided by the modal price.
    ``price_cols`` is ordered as (minimum, maximum, modal) and defaults to
    ``predictor_price_columns(df)``. Both features are missing where an input
    price is missing, and the ratio is missing where the modal price is not
    positive.
    """
    min_col, max_col, modal_col = price_cols or predictor_price_columns(df)
    out = df.copy()
    spread = (out[max_col] - out[min_col]).astype("float32")
    modal = out[modal_col].astype("float32")
    out[config.SPREAD_COLUMN] = spread
    out[config.SPREAD_RATIO_COLUMN] = (spread / modal.where(modal > 0)).astype(
        "float32"
    )
    return out


def add_calendar_features(
    df: pd.DataFrame, date_col: str = config.DATE_COLUMN
) -> pd.DataFrame:
    """Add month, year, day of year and season (features 15 to 18).

    The values come from the prediction date in ``date_col`` only.
    ``config.SEASON_COLUMN`` follows the documented IMD mapping in
    ``config.SEASON_BY_MONTH`` and is a categorical with the categories of
    ``config.SEASONS``, so every frame gets the same categories in the same
    order. All four features are missing where the date is missing.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    out[config.MONTH_COLUMN] = dates.dt.month.astype("Int8")
    out[config.YEAR_COLUMN] = dates.dt.year.astype("Int16")
    out[config.DAY_OF_YEAR_COLUMN] = dates.dt.dayofyear.astype("Int16")
    out[config.SEASON_COLUMN] = pd.Categorical(
        dates.dt.month.map(config.SEASON_BY_MONTH), categories=config.SEASONS
    )
    return out


def lag_column(days: int) -> str:
    """Return the name of the modal price lag feature for ``days``."""
    return config.LAG_COLUMN_TEMPLATE.format(days=days)


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    keys: list[str] | None = None,
    date_col: str = config.DATE_COLUMN,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Add modal price lags in calendar days (features 8 to 10).

    For each row, ``lag_column(n)`` holds the value of ``price_col`` on the
    date exactly ``n`` calendar days earlier in the same series, for every
    ``n`` in ``lags`` (default ``config.LAG_DAYS``). Series are identified by
    the ``keys`` present in ``df`` (default ``config.SERIES_KEYS``; missing
    key values form their own series). Lags are matched on dates, not row
    positions, so a gap in a series gives a missing lag instead of an older
    price, and the result does not depend on row order. ``price_col``
    defaults to the modal column of ``predictor_price_columns(df)``. Expects
    one row per series and date; if a date repeats, its first row is used as
    the lagged value. Rows keep their original order and index.
    """
    lag_days = config.LAG_DAYS if lags is None else lags
    source = price_col or predictor_price_columns(df)[2]
    key_cols = [k for k in (keys or config.SERIES_KEYS) if k in df.columns]
    out = df.copy()
    if df.empty:
        for days in lag_days:
            out[lag_column(days)] = pd.Series(dtype="float32")
        return out

    if key_cols:
        series = df.groupby(key_cols, observed=True, dropna=False, sort=False).ngroup()
    else:
        series = pd.Series(0, index=df.index)
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    lookup = pd.Series(
        df[source].to_numpy(dtype="float32"),
        index=pd.MultiIndex.from_arrays([series.to_numpy(), dates.to_numpy()]),
    )
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    for days in lag_days:
        past = pd.MultiIndex.from_arrays(
            [series.to_numpy(), (dates - pd.Timedelta(days=days)).to_numpy()]
        )
        out[lag_column(days)] = lookup.reindex(past).to_numpy(dtype="float32")
    return out
