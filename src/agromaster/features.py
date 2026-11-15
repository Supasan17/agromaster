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
    out = df.copy()
    for days in lag_days:
        out[lag_column(days)] = _lagged_values(df, days, keys, date_col, source)
    return out


def _series_ids(df: pd.DataFrame, keys: list[str] | None) -> np.ndarray:
    """Return an integer series id per row from the ``keys`` present in ``df``."""
    key_cols = [k for k in (keys or config.SERIES_KEYS) if k in df.columns]
    if not key_cols:
        return np.zeros(len(df), dtype="int64")
    return (
        df.groupby(key_cols, observed=True, dropna=False, sort=False)
        .ngroup()
        .to_numpy()
    )


def _lagged_values(
    df: pd.DataFrame,
    days: int,
    keys: list[str] | None,
    date_col: str,
    price_col: str,
) -> np.ndarray:
    """Return ``price_col`` exactly ``days`` calendar days earlier per series.

    Values are matched on (series, date), so a missing earlier date gives a
    missing value. If a date repeats within a series, its first row is used.
    """
    if df.empty:
        return np.empty(0, dtype="float32")
    series = _series_ids(df, keys)
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    lookup = pd.Series(
        df[price_col].to_numpy(dtype="float32"),
        index=pd.MultiIndex.from_arrays([series, dates.to_numpy()]),
    )
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    past = pd.MultiIndex.from_arrays(
        [series, (dates - pd.Timedelta(days=days)).to_numpy()]
    )
    return lookup.reindex(past).to_numpy(dtype="float32")


def ma_column(days: int) -> str:
    """Return the name of the rolling mean feature for a ``days`` window."""
    return config.MA_COLUMN_TEMPLATE.format(days=days)


def _rolling_stat(
    df: pd.DataFrame,
    window_days: int,
    stat: str,
    min_periods: int,
    keys: list[str] | None,
    date_col: str,
    price_col: str,
) -> np.ndarray:
    """Return a calendar-day rolling ``stat`` ("mean" or "std") per series.

    The window for a row on date ``t`` covers the dates ``t - window_days + 1``
    to ``t`` inclusive in the same series, so it never looks past ``t``.
    Missing prices inside the window are skipped; the result is missing when
    fewer than ``min_periods`` prices are observed. Rows on the same series
    and date share a window, so each sees all of that date's rows.
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
    rolling = (
        work.set_index("_date")
        .groupby("_series", sort=False)["_price"]
        .rolling(f"{window_days}D", min_periods=min_periods)
    )
    values = getattr(rolling, stat)().to_numpy()
    # A time-based window ends at the current row, so rows sharing a date
    # would otherwise see different windows; give each the last row's value.
    keyed = pd.Series(values, index=pd.MultiIndex.from_frame(work[["_series", "_date"]]))
    values = keyed.groupby(level=[0, 1], sort=False).transform("last").to_numpy()
    result = np.full(len(df), np.nan, dtype="float32")
    result[work["_row"].to_numpy()] = values
    return result


def add_rolling_features(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    date_col: str = config.DATE_COLUMN,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Add rolling mean, volatility and momentum features (features 11 to 14).

    Within each series (``keys`` present in ``df``, default
    ``config.SERIES_KEYS``), and using calendar-day windows that end on the
    prediction date:

    * ``ma_column(n)`` for ``n`` in ``config.MA_WINDOWS`` is the mean of
      ``price_col`` over the last ``n`` days (at least
      ``config.MA_MIN_PERIODS`` observed prices);
    * ``config.VOLATILITY_COLUMN`` is the sample standard deviation over the
      last ``config.VOLATILITY_WINDOW`` days (at least
      ``config.VOLATILITY_MIN_PERIODS`` observed prices);
    * ``config.MOMENTUM_COLUMN`` is ``100 * (p_t - p_{t-k}) / p_{t-k}`` with
      ``k = config.MOMENTUM_DAYS`` calendar days, missing when either price
      is missing or the earlier price is not positive.

    ``price_col`` defaults to the modal column of
    ``predictor_price_columns(df)``. Only prices on or before each row's date
    are used, the result does not depend on row order, and rows keep their
    original order and index.
    """
    source = price_col or predictor_price_columns(df)[2]
    out = df.copy()
    for days in config.MA_WINDOWS:
        out[ma_column(days)] = _rolling_stat(
            df, days, "mean", config.MA_MIN_PERIODS, keys, date_col, source
        )
    out[config.VOLATILITY_COLUMN] = _rolling_stat(
        df,
        config.VOLATILITY_WINDOW,
        "std",
        config.VOLATILITY_MIN_PERIODS,
        keys,
        date_col,
        source,
    )
    current = df[source].to_numpy(dtype="float32")
    past = _lagged_values(df, config.MOMENTUM_DAYS, keys, date_col, source)
    past = np.where(past > 0, past, np.nan)
    out[config.MOMENTUM_COLUMN] = (100 * (current - past) / past).astype("float32")
    return out
