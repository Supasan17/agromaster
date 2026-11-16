"""Tests for the 7-day ahead ratio and risk thresholds in agromaster.labels."""
import numpy as np
import pandas as pd
import pytest

from agromaster import config, labels

MODAL = labels.MODAL_COLUMN


def make_series(
    start: str, modal: list[float], commodity: str = "Onion", market: str = "Pune"
) -> pd.DataFrame:
    """Daily one-series frame with the given observed modal prices."""
    return pd.DataFrame(
        {
            "Market": market,
            "Commodity": commodity,
            config.DATE_COLUMN: pd.date_range(start, periods=len(modal), freq="D"),
            MODAL: np.array(modal, dtype="float32"),
        }
    )


def make_ratios(commodity: str, ratios: list[float]) -> pd.DataFrame:
    """Frame holding only a commodity column and precomputed ratios."""
    return pd.DataFrame(
        {
            config.COMMODITY_COLUMN: commodity,
            config.RATIO_COLUMN: np.array(ratios, dtype="float64"),
        }
    )


# 7-day ahead price ratio


def test_future_price_is_horizon_days_ahead() -> None:
    df = make_series("2024-01-01", [float(100 + i) for i in range(10)])
    out = labels.add_future_ratio(df, horizon_days=7, history_days=3)
    future = out[config.FUTURE_PRICE_COLUMN]
    assert future.iloc[:3].tolist() == [107.0, 108.0, 109.0]
    assert future.iloc[3:].isna().all()


def test_history_mean_excludes_prediction_date() -> None:
    df = make_series("2024-01-01", [10.0, 20.0, 30.0, 40.0, 1000.0])
    out = labels.add_future_ratio(df, horizon_days=1, history_days=3)
    history = out[config.HISTORY_MEAN_COLUMN]
    assert np.isnan(history.iloc[0])
    assert history.iloc[1:].tolist() == pytest.approx([10.0, 15.0, 20.0, 30.0])


def test_ratio_value_on_known_series() -> None:
    df = make_series("2024-01-01", [100.0, 100.0, 100.0, 80.0, 90.0, 120.0])
    out = labels.add_future_ratio(df, horizon_days=2, history_days=3)
    ratio = out[config.RATIO_COLUMN]
    # t = day 3: future = modal(day 5) = 120, history = mean(100, 100, 100).
    assert ratio.iloc[3] == pytest.approx(1.2)
    # t = day 2: future = modal(day 4) = 90, history = mean(100, 100).
    assert ratio.iloc[2] == pytest.approx(0.9)
    assert ratio.dtype == "float32"


def test_default_window_lengths_follow_config() -> None:
    n = config.HISTORY_DAYS + config.HORIZON_DAYS + 1
    df = make_series("2023-01-01", [100.0] * (n - 1) + [50.0])
    out = labels.add_future_ratio(df)
    t = n - 1 - config.HORIZON_DAYS
    assert out[config.FUTURE_PRICE_COLUMN].iloc[t] == pytest.approx(50.0)
    assert out[config.RATIO_COLUMN].iloc[t] == pytest.approx(0.5)


def test_missing_future_day_is_not_filled() -> None:
    df = make_series("2024-01-01", [100.0] * 6)
    df = df.drop(index=4).reset_index(drop=True)  # no report on day 4
    out = labels.add_future_ratio(df, horizon_days=2, history_days=3)
    day2 = out[config.DATE_COLUMN] == pd.Timestamp("2024-01-03")
    assert out.loc[day2, config.FUTURE_PRICE_COLUMN].isna().all()
    assert out.loc[day2, config.RATIO_COLUMN].isna().all()


def test_missing_observed_price_gives_missing_future() -> None:
    df = make_series("2024-01-01", [100.0, 100.0, 100.0, np.nan, 100.0])
    out = labels.add_future_ratio(df, horizon_days=2, history_days=2)
    assert np.isnan(out[config.FUTURE_PRICE_COLUMN].iloc[1])
    assert np.isnan(out[config.RATIO_COLUMN].iloc[1])


def test_non_positive_history_gives_missing_ratio() -> None:
    df = make_series("2024-01-01", [0.0, 0.0, 100.0, 100.0])
    out = labels.add_future_ratio(df, horizon_days=1, history_days=2)
    assert np.isnan(out[config.RATIO_COLUMN].iloc[2])


def test_ratio_stays_within_series() -> None:
    a = make_series("2024-01-01", [100.0, 100.0, 100.0], market="Pune")
    b = make_series("2024-01-01", [500.0, 500.0, 500.0], market="Nashik")
    out = labels.add_future_ratio(pd.concat([a, b], ignore_index=True), horizon_days=1)
    assert out[config.RATIO_COLUMN].iloc[[1, 4]].tolist() == pytest.approx([1.0, 1.0])
    assert out[config.FUTURE_PRICE_COLUMN].iloc[[0, 3]].tolist() == [100.0, 500.0]


def test_ratio_does_not_depend_on_row_order() -> None:
    df = make_series("2024-01-01", [float(100 + 5 * i) for i in range(12)])
    shuffled = df.sample(frac=1.0, random_state=config.SEED)
    out = labels.add_future_ratio(df, horizon_days=3, history_days=5)
    out_shuffled = labels.add_future_ratio(shuffled, horizon_days=3, history_days=5)
    assert out_shuffled.index.tolist() == shuffled.index.tolist()
    pd.testing.assert_frame_equal(out_shuffled.sort_index(), out)


def test_add_future_ratio_leaves_input_unchanged() -> None:
    df = make_series("2024-01-01", [100.0, 110.0, 120.0])
    before = df.copy()
    labels.add_future_ratio(df, horizon_days=1, history_days=2)
    pd.testing.assert_frame_equal(df, before)


# Fold-fitted commodity risk thresholds


def test_thresholds_are_commodity_quantiles() -> None:
    train = pd.concat(
        [
            make_ratios("Onion", [float(i) for i in range(1, 11)]),
            make_ratios("Potato", [float(i) for i in range(101, 111)]),
        ],
        ignore_index=True,
    )
    fitted = labels.fit_risk_thresholds(train).set_index(config.COMMODITY_COLUMN)
    onion = np.arange(1, 11, dtype="float64")
    potato = np.arange(101, 111, dtype="float64")
    assert fitted.loc["Onion", config.HIGH_THRESHOLD_COLUMN] == pytest.approx(
        np.quantile(onion, config.HIGH_RISK_QUANTILE)
    )
    assert fitted.loc["Onion", config.MEDIUM_THRESHOLD_COLUMN] == pytest.approx(
        np.quantile(onion, config.MEDIUM_RISK_QUANTILE)
    )
    assert fitted.loc["Potato", config.HIGH_THRESHOLD_COLUMN] == pytest.approx(
        np.quantile(potato, config.HIGH_RISK_QUANTILE)
    )
    assert fitted.index.tolist() == ["Onion", "Potato"]


def test_thresholds_ignore_missing_and_infinite_ratios() -> None:
    clean = make_ratios("Onion", [1.0, 2.0, 3.0, 4.0])
    noisy = pd.concat(
        [clean, make_ratios("Onion", [np.nan, np.inf, -np.inf])], ignore_index=True
    )
    pd.testing.assert_frame_equal(
        labels.fit_risk_thresholds(noisy), labels.fit_risk_thresholds(clean)
    )


def test_thresholds_skip_groups_without_ratios() -> None:
    train = pd.concat(
        [make_ratios("Onion", [1.0, 2.0]), make_ratios("Garlic", [np.nan])],
        ignore_index=True,
    )
    fitted = labels.fit_risk_thresholds(train)
    assert fitted[config.COMMODITY_COLUMN].tolist() == ["Onion"]


@pytest.mark.parametrize("high, medium", [(0.3, 0.1), (0.0, 0.3), (0.1, 1.0), (0.2, 0.2)])
def test_invalid_quantiles_raise(high: float, medium: float) -> None:
    with pytest.raises(ValueError, match="Quantiles"):
        labels.fit_risk_thresholds(
            make_ratios("Onion", [1.0, 2.0]), high_quantile=high, medium_quantile=medium
        )


def test_apply_labels_by_threshold_with_inclusive_bounds() -> None:
    thresholds = pd.DataFrame(
        {
            config.COMMODITY_COLUMN: ["Onion"],
            config.HIGH_THRESHOLD_COLUMN: [0.8],
            config.MEDIUM_THRESHOLD_COLUMN: [0.95],
        }
    )
    df = make_ratios("Onion", [0.5, 0.8, 0.9, 0.95, 1.2])
    out = labels.apply_risk_thresholds(df, thresholds)
    assert out[config.LABEL_COLUMN].tolist() == ["High", "High", "Medium", "Medium", "Low"]
    assert out[config.LABEL_COLUMN].cat.categories.tolist() == config.RISK_LABELS
    assert out[config.LABEL_COLUMN].cat.ordered


def test_apply_leaves_label_missing_for_unknown_commodity_or_ratio() -> None:
    thresholds = labels.fit_risk_thresholds(make_ratios("Onion", [1.0, 2.0, 3.0]))
    df = pd.concat(
        [make_ratios("Onion", [np.nan, 0.5]), make_ratios("Garlic", [0.5])],
        ignore_index=True,
    )
    out = labels.apply_risk_thresholds(df, thresholds)
    assert out[config.LABEL_COLUMN].isna().tolist() == [True, False, True]


def test_thresholds_are_applied_unchanged_to_other_frame() -> None:
    train = make_ratios("Onion", [float(i) for i in range(1, 101)])
    thresholds = labels.fit_risk_thresholds(train)
    before = thresholds.copy()
    # A shifted validation frame must be labelled with the training cut-offs,
    # not with thresholds refitted on itself.
    valid = make_ratios("Onion", [1000.0, 1001.0, 1002.0])
    out = labels.apply_risk_thresholds(valid, thresholds)
    assert (out[config.LABEL_COLUMN] == "Low").all()
    pd.testing.assert_frame_equal(thresholds, before)


def test_training_label_shares_follow_quantiles() -> None:
    train = make_ratios("Onion", [float(i) for i in range(1, 101)])
    out = labels.apply_risk_thresholds(train, labels.fit_risk_thresholds(train))
    shares = out[config.LABEL_COLUMN].value_counts(normalize=True)
    assert shares["High"] == pytest.approx(config.HIGH_RISK_QUANTILE, abs=0.02)
    assert shares["Medium"] == pytest.approx(
        config.MEDIUM_RISK_QUANTILE - config.HIGH_RISK_QUANTILE, abs=0.02
    )


def test_apply_keeps_order_index_and_input() -> None:
    thresholds = labels.fit_risk_thresholds(make_ratios("Onion", [1.0, 2.0, 3.0]))
    df = make_ratios("Onion", [3.0, 1.0, 2.0])
    df.index = [10, 5, 7]
    before = df.copy()
    out = labels.apply_risk_thresholds(df, thresholds)
    assert out.index.tolist() == [10, 5, 7]
    pd.testing.assert_frame_equal(df, before)
