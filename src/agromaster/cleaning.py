"""Record-level cleaning of the interim AGMARKNET price table.

Follows the preprocessing list in the proposal: duplicate records, records
with missing or non-positive prices, and records whose prices break the
minimum <= modal <= maximum ordering are removed before any aggregation,
alignment or feature engineering. Category names are then standardised so
that the same product or place is not split across spellings, and records
of one series reported more than once on the same date are merged with the
median of each price field, and each series is aligned to consecutive
calendar days with a short, predictor-only forward-fill. Each step is a
small function that takes and returns a pandas DataFrame and leaves its input
unchanged.
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


def standardise_name(value: str, aliases: dict[str, str] | None = None) -> str:
    """Return the canonical spelling of one category name.

    The value is trimmed, runs of whitespace become one space and the result
    is case-folded. A case-folded form found in ``aliases`` maps to its alias;
    any other name is returned in title case.
    """
    key = " ".join(value.split()).casefold()
    return (aliases or {}).get(key, key.title())


def standardise_names(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    aliases: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Standardise category names so one product or place has one spelling.

    Only ``columns`` present in ``df`` are changed; missing values stay
    missing and categorical columns stay categorical. ``aliases`` defaults to
    ``config.NAME_ALIASES`` and holds one alias map per column.
    """
    alias_maps = config.NAME_ALIASES if aliases is None else aliases
    out = df.copy()
    for col in columns or config.NAME_COLUMNS:
        if col not in out.columns:
            continue
        col_aliases = alias_maps.get(col, {})
        values = out[col]
        if isinstance(values.dtype, pd.CategoricalDtype):
            mapping = {
                c: standardise_name(str(c), col_aliases) for c in values.cat.categories
            }
            out[col] = values.astype(object).map(mapping).astype("category")
        else:
            out[col] = values.map(
                lambda v: standardise_name(str(v), col_aliases) if pd.notna(v) else v
            )
    return out


def aggregate_same_day(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    date_col: str | None = None,
    price_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge records of one series on one date into a single record.

    Records are grouped by the series ``keys`` present in ``df`` (default
    ``config.SERIES_KEYS``) and the date column, and each price field is
    replaced by its median within the group. Missing key values form their
    own group rather than being dropped. Because the median keeps order, rows
    that already satisfy min <= modal <= max still do after aggregation.
    Columns other than the keys, the date and the prices are dropped. The
    result is sorted by series and date.
    """
    date = date_col or config.DATE_COLUMN
    prices = price_cols or config.PRICE_COLUMNS
    group_cols = [k for k in (keys or config.SERIES_KEYS) if k in df.columns] + [date]
    return (
        df.groupby(group_cols, observed=True, dropna=False, sort=True)[prices]
        .median()
        .reset_index()
    )


def align_calendar_days(
    df: pd.DataFrame,
    keys: list[str] | None = None,
    date_col: str | None = None,
    price_cols: list[str] | None = None,
    max_ffill_days: int | None = None,
) -> pd.DataFrame:
    """Reindex each series to consecutive calendar days.

    Expects one record per series and date (see ``aggregate_same_day``). Each
    series, identified by the ``keys`` present in ``df`` (missing key values
    form their own series), gets one row for every calendar day from its
    first to its last observed date. The original price columns keep only
    observed values, so days without a report stay missing and future target
    prices are never filled. For every price column a predictor copy named
    ``<col>`` + ``config.FILLED_SUFFIX`` carries the last observed value
    forward for at most ``max_ffill_days`` days (default
    ``config.MAX_FFILL_DAYS``); days further from the last report stay
    missing. ``config.OBSERVED_COLUMN`` flags days with an actual report. The
    result is sorted by series and date.
    """
    date = date_col or config.DATE_COLUMN
    prices = price_cols or config.PRICE_COLUMNS
    limit = config.MAX_FFILL_DAYS if max_ffill_days is None else max_ffill_days
    key_cols = [k for k in (keys or config.SERIES_KEYS) if k in df.columns]
    filled_cols = [c + config.FILLED_SUFFIX for c in prices]
    if df.empty:
        out = df[key_cols + [date] + prices].copy()
        out[config.OBSERVED_COLUMN] = pd.Series(dtype=bool)
        for col, filled in zip(prices, filled_cols):
            out[filled] = out[col]
        return out.reset_index(drop=True)

    obs = df[key_cols + [date] + prices].copy()
    obs[date] = pd.to_datetime(obs[date]).dt.normalize()
    if key_cols:
        obs["_series"] = obs.groupby(
            key_cols, observed=True, dropna=False, sort=True
        ).ngroup()
    else:
        obs["_series"] = 0

    spans = obs.groupby("_series")[date].agg(["min", "max"])
    lengths = (spans["max"] - spans["min"]).dt.days.to_numpy() + 1
    series_ids = spans.index.repeat(lengths)
    day_offsets = pd.Series(1, index=series_ids).groupby(level=0).cumsum() - 1
    grid = pd.DataFrame(
        {
            "_series": series_ids,
            date: spans["min"].reindex(series_ids).to_numpy()
            + pd.to_timedelta(day_offsets.to_numpy(), unit="D"),
        }
    )

    out = grid.merge(obs[["_series", date] + prices], on=["_series", date], how="left")
    out[config.OBSERVED_COLUMN] = out[prices].notna().any(axis=1)
    filled = out.groupby("_series", sort=False)[prices].ffill(limit=limit)
    for col, filled_col in zip(prices, filled_cols):
        out[filled_col] = filled[col]

    if key_cols:
        series_keys = obs.drop_duplicates("_series").set_index("_series")[key_cols]
        out = out.join(series_keys, on="_series")
    out = out.sort_values(["_series", date], kind="stable")
    ordered = key_cols + [date] + prices + [config.OBSERVED_COLUMN] + filled_cols
    return out[ordered].reset_index(drop=True)
