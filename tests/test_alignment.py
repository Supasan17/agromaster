"""Tests for same-day aggregation and calendar alignment in agromaster.cleaning."""
import numpy as np
import pandas as pd

from agromaster import cleaning, config

FILLED = [c + config.FILLED_SUFFIX for c in config.PRICE_COLUMNS]


def make_series(dates: list[str], modal: list[float], market: str = "Pune") -> pd.DataFrame:
    """One-commodity series with min/max set around the modal price."""
    modal_arr = np.array(modal, dtype=float)
    return pd.DataFrame(
        {
            "Market": market,
            "Commodity": "Onion",
            config.DATE_COLUMN: pd.to_datetime(dates),
            "Min_Price": modal_arr - 100.0,
            "Max_Price": modal_arr + 100.0,
            "Modal_Price": modal_arr,
        }
    )


def test_aggregate_same_day_takes_median_per_series_and_date() -> None:
    df = pd.concat(
        [
            make_series(["2024-01-01"] * 3, [1000.0, 1200.0, 1700.0]),
            make_series(["2024-01-02"], [1500.0]),
            make_series(["2024-01-01"], [900.0], market="Nashik"),
        ],
        ignore_index=True,
    )
    df["Notes"] = "x"
    out = cleaning.aggregate_same_day(df)
    assert len(out) == 3
    assert "Notes" not in out.columns
    pune = out[(out["Market"] == "Pune") & (out[config.DATE_COLUMN] == "2024-01-01")]
    assert pune["Modal_Price"].iloc[0] == 1200.0
    assert pune["Min_Price"].iloc[0] == 1100.0
    assert pune["Max_Price"].iloc[0] == 1300.0
    assert out["Market"].tolist() == ["Nashik", "Pune", "Pune"]
    assert len(df) == 5


def test_aggregate_same_day_keeps_price_order() -> None:
    df = pd.DataFrame(
        {
            "Market": ["Pune"] * 3,
            config.DATE_COLUMN: pd.to_datetime(["2024-01-01"] * 3),
            "Min_Price": [100.0, 400.0, 150.0],
            "Max_Price": [300.0, 900.0, 200.0],
            "Modal_Price": [200.0, 500.0, 180.0],
        }
    )
    out = cleaning.aggregate_same_day(df)
    row = out.iloc[0]
    assert row["Min_Price"] <= row["Modal_Price"] <= row["Max_Price"]


def test_aggregate_same_day_keeps_missing_keys_as_group() -> None:
    df = make_series(["2024-01-01", "2024-01-01"], [1000.0, 2000.0])
    df["Variety"] = [None, None]
    out = cleaning.aggregate_same_day(df)
    assert len(out) == 1
    assert out["Modal_Price"].iloc[0] == 1500.0


def test_align_calendar_days_fills_gaps_per_series() -> None:
    df = pd.concat(
        [
            make_series(["2024-01-01", "2024-01-04"], [1000.0, 1300.0]),
            make_series(["2024-01-02", "2024-01-03"], [500.0, 600.0], market="Nashik"),
        ],
        ignore_index=True,
    )
    out = cleaning.align_calendar_days(df)
    pune = out[out["Market"] == "Pune"]
    nashik = out[out["Market"] == "Nashik"]
    assert pune[config.DATE_COLUMN].dt.strftime("%m-%d").tolist() == [
        "01-01",
        "01-02",
        "01-03",
        "01-04",
    ]
    assert len(nashik) == 2
    assert pune[config.OBSERVED_COLUMN].tolist() == [True, False, False, True]
    assert out.columns.tolist()[-len(FILLED):] == FILLED


def test_align_calendar_days_never_fills_target_prices() -> None:
    out = cleaning.align_calendar_days(
        make_series(["2024-01-01", "2024-01-03"], [1000.0, 1100.0])
    )
    assert pd.isna(out["Modal_Price"].iloc[1])
    assert out["Modal_Price" + config.FILLED_SUFFIX].iloc[1] == 1000.0


def test_align_calendar_days_limits_forward_fill() -> None:
    out = cleaning.align_calendar_days(
        make_series(["2024-01-01", "2024-01-07"], [1000.0, 1600.0]), max_ffill_days=2
    )
    filled = out["Modal_Price" + config.FILLED_SUFFIX]
    assert filled.iloc[1:3].tolist() == [1000.0, 1000.0]
    assert filled.iloc[3:6].isna().all()
    assert filled.iloc[6] == 1600.0


def test_align_calendar_days_default_limit_from_config() -> None:
    n_gap = config.MAX_FFILL_DAYS + 2
    dates = pd.date_range("2024-01-01", periods=n_gap + 2).strftime("%Y-%m-%d")
    out = cleaning.align_calendar_days(make_series([dates[0], dates[-1]], [1000.0, 2000.0]))
    filled = out["Modal_Price" + config.FILLED_SUFFIX]
    assert filled.iloc[1 : config.MAX_FFILL_DAYS + 1].eq(1000.0).all()
    assert pd.isna(filled.iloc[config.MAX_FFILL_DAYS + 1])


def test_align_calendar_days_does_not_fill_across_series() -> None:
    df = pd.concat(
        [
            make_series(["2024-01-01"], [1000.0], market="Akola"),
            make_series(["2024-01-01", "2024-01-03"], [500.0, 700.0], market="Pune"),
        ],
        ignore_index=True,
    )
    out = cleaning.align_calendar_days(df)
    pune = out[out["Market"] == "Pune"]
    assert pune["Modal_Price" + config.FILLED_SUFFIX].tolist() == [500.0, 500.0, 700.0]
    assert len(out[out["Market"] == "Akola"]) == 1


def test_align_calendar_days_handles_empty_frame() -> None:
    empty = make_series([], [])
    out = cleaning.align_calendar_days(empty)
    assert out.empty
    assert config.OBSERVED_COLUMN in out.columns
    assert set(FILLED) <= set(out.columns)


def test_aggregate_then_align_leaves_input_unchanged() -> None:
    df = make_series(["2024-01-01", "2024-01-01", "2024-01-03"], [1000.0, 1400.0, 900.0])
    before = df.copy()
    out = cleaning.align_calendar_days(cleaning.aggregate_same_day(df))
    assert out["Modal_Price"].iloc[0] == 1200.0
    assert len(out) == 3
    pd.testing.assert_frame_equal(df, before)
