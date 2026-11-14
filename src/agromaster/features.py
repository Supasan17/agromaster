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
