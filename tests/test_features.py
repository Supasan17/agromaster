"""Tests for feature correctness and leakage in agromaster.features."""
import numpy as np
import pandas as pd
import pytest

from agromaster import cleaning, config, features

FILLED = [c + config.FILLED_SUFFIX for c in config.PRICE_COLUMNS]
MODAL = "Modal_Price"
FEATURE_COLUMNS = (
    [config.SPREAD_COLUMN, config.SPREAD_RATIO_COLUMN]
    + [features.lag_column(d) for d in config.LAG_DAYS]
    + [features.ma_column(d) for d in config.MA_WINDOWS]
    + [config.VOLATILITY_COLUMN, config.MOMENTUM_COLUMN]
)


def make_series(
    start: str, modal: list[float], market: str = "Pune"
) -> pd.DataFrame:
    """Daily one-commodity series with min/max set around the modal price."""
    modal_arr = np.array(modal, dtype="float32")
    return pd.DataFrame(
        {
            "Market": market,
            "Commodity": "Onion",
            config.DATE_COLUMN: pd.date_range(start, periods=len(modal), freq="D"),
            "Min_Price": modal_arr - 100,
            "Max_Price": modal_arr + 100,
            MODAL: modal_arr,
        }
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature step in pipeline order."""
    out = features.add_price_spread(df)
    out = features.add_calendar_features(out)
    out = features.add_lag_features(out)
    return features.add_rolling_features(out)


# Price spread and spread ratio (features 6 and 7)


def test_price_spread_and_ratio_values() -> None:
    df = make_series("2024-01-01", [1000.0, 500.0])
    out = features.add_price_spread(df)
    assert out[config.SPREAD_COLUMN].tolist() == [200.0, 200.0]
    assert out[config.SPREAD_RATIO_COLUMN].tolist() == pytest.approx([0.2, 0.4])
    assert out[config.SPREAD_COLUMN].dtype == "float32"
    assert config.SPREAD_COLUMN not in df.columns


def test_spread_ratio_missing_for_non_positive_or_missing_modal() -> None:
    df = make_series("2024-01-01", [1000.0, 0.0, 800.0])
    df.loc[2, "Min_Price"] = np.nan
    out = features.add_price_spread(df)
    assert np.isnan(out[config.SPREAD_RATIO_COLUMN].iloc[1])
    assert np.isnan(out[config.SPREAD_COLUMN].iloc[2])
    assert np.isnan(out[config.SPREAD_RATIO_COLUMN].iloc[2])


def test_predictor_columns_prefer_filled_copies() -> None:
    df = make_series("2024-01-01", [1000.0])
    assert features.predictor_price_columns(df) == config.PRICE_COLUMNS
    for col, filled in zip(config.PRICE_COLUMNS, FILLED):
        df[filled] = df[col]
    assert features.predictor_price_columns(df) == FILLED


# Calendar and season features (features 15 to 18)


def test_calendar_features_and_season_mapping() -> None:
    dates = pd.to_datetime(["2024-01-15", "2024-04-01", "2024-07-31", "2024-12-31"])
    df = pd.DataFrame({config.DATE_COLUMN: dates})
    out = features.add_calendar_features(df)
    assert out[config.MONTH_COLUMN].tolist() == [1, 4, 7, 12]
    assert out[config.YEAR_COLUMN].tolist() == [2024] * 4
    assert out[config.DAY_OF_YEAR_COLUMN].tolist() == [15, 92, 213, 366]
    assert out[config.SEASON_COLUMN].tolist() == [
        "Winter", "Summer", "Monsoon", "Post-monsoon"
    ]
    assert list(out[config.SEASON_COLUMN].cat.categories) == config.SEASONS


def test_season_mapping_covers_every_month() -> None:
    assert sorted(config.SEASON_BY_MONTH) == list(range(1, 13))
    assert set(config.SEASON_BY_MONTH.values()) == set(config.SEASONS)


def test_calendar_features_missing_for_missing_date() -> None:
    df = pd.DataFrame({config.DATE_COLUMN: pd.to_datetime(["2024-03-01", None])})
    out = features.add_calendar_features(df)
    assert out[config.MONTH_COLUMN].isna().tolist() == [False, True]
    assert pd.isna(out[config.SEASON_COLUMN].iloc[1])


# Lag features (features 8 to 10)


def test_lags_use_calendar_days_within_series() -> None:
    df = make_series("2024-01-01", [float(100 + i) for i in range(40)])
    out = features.add_lag_features(df)
    assert np.isnan(out[features.lag_column(1)].iloc[0])
    assert out[features.lag_column(1)].iloc[5] == 104.0
    assert out[features.lag_column(7)].iloc[10] == 103.0
    assert np.isnan(out[features.lag_column(30)].iloc[29])
    assert out[features.lag_column(30)].iloc[35] == 105.0


def test_lag_gap_gives_missing_not_older_price() -> None:
    df = make_series("2024-01-01", [100.0, 110.0, 120.0])
    df = df.drop(index=1).reset_index(drop=True)
    out = features.add_lag_features(df, lags=[1])
    assert np.isnan(out[features.lag_column(1)].iloc[1])


def test_lags_do_not_cross_series() -> None:
    df = pd.concat(
        [
            make_series("2024-01-01", [100.0, 110.0], market="Pune"),
            make_series("2024-01-02", [900.0], market="Nashik"),
        ],
        ignore_index=True,
    )
    out = features.add_lag_features(df, lags=[1])
    assert out[features.lag_column(1)].iloc[1] == 100.0
    assert np.isnan(out[features.lag_column(1)].iloc[2])


def test_lags_do_not_depend_on_row_order() -> None:
    df = make_series("2024-01-01", [float(100 + i) for i in range(10)])
    shuffled = df.sample(frac=1, random_state=config.SEED)
    out = features.add_lag_features(shuffled, lags=[1, 7])
    expected = features.add_lag_features(df, lags=[1, 7])
    pd.testing.assert_frame_equal(out.sort_index(), expected)


# Rolling mean, volatility and momentum (features 11 to 14)


def test_rolling_mean_volatility_and_momentum_values() -> None:
    modal = [float(100 + 10 * i) for i in range(35)]
    out = features.add_rolling_features(make_series("2024-01-01", modal))
    t = 30
    assert out[features.ma_column(7)].iloc[t] == pytest.approx(np.mean(modal[t - 6 : t + 1]))
    assert out[features.ma_column(30)].iloc[t] == pytest.approx(
        np.mean(modal[t - 29 : t + 1])
    )
    assert out[config.VOLATILITY_COLUMN].iloc[t] == pytest.approx(
        np.std(modal[t - 29 : t + 1], ddof=1), rel=1e-5
    )
    expected_momentum = 100 * (modal[t] - modal[t - 7]) / modal[t - 7]
    assert out[config.MOMENTUM_COLUMN].iloc[t] == pytest.approx(expected_momentum, rel=1e-5)


def test_rolling_min_periods_and_first_rows() -> None:
    out = features.add_rolling_features(make_series("2024-01-01", [100.0, 120.0]))
    assert out[features.ma_column(7)].iloc[0] == 100.0
    assert np.isnan(out[config.VOLATILITY_COLUMN].iloc[0])
    assert not np.isnan(out[config.VOLATILITY_COLUMN].iloc[1])
    assert out[config.MOMENTUM_COLUMN].isna().all()


def test_momentum_missing_for_non_positive_past_price() -> None:
    modal = [0.0] + [100.0] * 7
    out = features.add_rolling_features(make_series("2024-01-01", modal))
    assert np.isnan(out[config.MOMENTUM_COLUMN].iloc[7])


def test_rolling_window_skips_missing_dates() -> None:
    df = make_series("2024-01-01", [100.0, 200.0, 300.0, 400.0])
    df = df.drop(index=[1, 2]).reset_index(drop=True)
    out = features.add_rolling_features(df)
    assert out[features.ma_column(7)].tolist() == pytest.approx([100.0, 250.0])


# Leakage: predictors must only use information up to the prediction date


@pytest.mark.parametrize("cut", [10, 25, 40])
def test_features_unchanged_when_future_prices_change(cut: int) -> None:
    rng = np.random.default_rng(config.SEED)
    modal = (1000 + rng.normal(0, 50, 60)).round(1).tolist()
    df = make_series("2024-01-01", modal)
    changed = df.copy()
    future = changed.index > cut
    for col in config.PRICE_COLUMNS:
        changed.loc[future, col] = changed.loc[future, col] * 3
    base = build_features(df)
    other = build_features(changed)
    pd.testing.assert_frame_equal(
        base.loc[: cut, FEATURE_COLUMNS], other.loc[: cut, FEATURE_COLUMNS]
    )


def test_features_unchanged_when_future_rows_are_removed() -> None:
    modal = [float(100 + (i % 9) * 7) for i in range(50)]
    df = make_series("2024-01-01", modal)
    full = build_features(df)
    truncated = build_features(df.iloc[:30])
    pd.testing.assert_frame_equal(
        full.loc[:29, FEATURE_COLUMNS], truncated[FEATURE_COLUMNS]
    )


def test_aligned_features_never_use_future_target_prices() -> None:
    df = make_series("2024-01-01", [100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    df = df.drop(index=[2, 3]).reset_index(drop=True)
    aligned = cleaning.align_calendar_days(df, max_ffill_days=1)
    out = build_features(aligned)
    # Day 3 is forward-filled from day 2; day 4 is beyond the fill limit, so
    # its predictors stay missing rather than reading the later price.
    day3 = out[out[config.DATE_COLUMN] == "2024-01-03"].iloc[0]
    day4 = out[out[config.DATE_COLUMN] == "2024-01-04"].iloc[0]
    assert np.isnan(day3[MODAL])
    assert day3[features.lag_column(1)] == 110.0
    assert day3[config.SPREAD_COLUMN] == 200.0
    assert np.isnan(day4[config.SPREAD_COLUMN])
    assert day4[features.ma_column(7)] == pytest.approx(np.mean([100.0, 110.0, 110.0]))


def test_feature_steps_leave_input_unchanged() -> None:
    df = make_series("2024-01-01", [100.0, 110.0, 120.0])
    before = df.copy()
    build_features(df)
    pd.testing.assert_frame_equal(df, before)
