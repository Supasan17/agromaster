"""Smoke tests for the encoders, baselines and classifiers in agromaster.models.

Every test builds a small synthetic frame with the model input columns, so no
real data is needed. Model sizes are kept tiny to make the tests fast.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from agromaster import config, models
from agromaster.labels import fit_risk_thresholds

LABEL = config.LABEL_COLUMN
COMMODITIES = ["Onion", "Potato", "Tomato"]
STATES = ["Kerala", "Punjab"]


def make_frame(n: int = 90, seed: int = 0) -> pd.DataFrame:
    """Frame with every model input column and a learnable risk label.

    The label depends on the 7-day momentum, so the models have a signal to
    find; every risk class occurs and the classes are imbalanced.
    """
    rng = np.random.default_rng(seed)
    numerical, _ = models.feature_columns(pd.DataFrame())
    df = pd.DataFrame(
        rng.normal(100.0, 10.0, size=(n, len(numerical))), columns=numerical
    )
    momentum = rng.normal(0.0, 1.0, size=n)
    df[config.MOMENTUM_COLUMN] = momentum
    df[config.COMMODITY_COLUMN] = rng.choice(COMMODITIES, size=n)
    df["State"] = rng.choice(STATES, size=n)
    df[config.SEASON_COLUMN] = rng.choice(config.SEASONS, size=n)
    labels = np.where(
        momentum < -0.9, "High", np.where(momentum < -0.3, "Medium", "Low")
    )
    df[LABEL] = pd.Categorical(labels, categories=config.RISK_LABELS, ordered=True)
    return df


def assert_valid_predictions(pred: pd.Series, df: pd.DataFrame) -> None:
    """Predictions cover every row of ``df`` with labels from RISK_LABELS."""
    assert pred.index.equals(df.index)
    assert pred.name == LABEL
    assert list(pred.cat.categories) == config.RISK_LABELS
    assert pred.notna().all()


# Feature columns


def test_feature_columns_follow_config() -> None:
    numerical, categorical = models.feature_columns(pd.DataFrame())
    assert numerical == config.NUMERICAL_FEATURES
    assert categorical == config.CATEGORICAL_FEATURES


def test_feature_columns_use_filled_prices_when_present() -> None:
    filled = [c + config.FILLED_SUFFIX for c in config.PRICE_COLUMNS]
    numerical, _ = models.feature_columns(pd.DataFrame(columns=filled))
    assert numerical[:3] == filled
    assert not set(config.PRICE_COLUMNS) & set(numerical)


# Encoders


def test_tree_encoder_codes_unseen_and_missing_categories() -> None:
    train = make_frame()
    test = make_frame(5, seed=1)
    test.loc[0, config.COMMODITY_COLUMN] = "Garlic"
    test.loc[1, "State"] = None
    encoder = models.fit_encoder(train, kind="tree")
    out = models.encode(encoder, test)

    numerical, categorical = models.feature_columns(train)
    assert list(out.columns) == numerical + categorical
    assert out.index.equals(test.index)
    assert out.loc[0, config.COMMODITY_COLUMN] == config.UNKNOWN_CATEGORY_CODE
    assert out.loc[1, "State"] == config.UNKNOWN_CATEGORY_CODE
    assert (out.loc[2:, config.COMMODITY_COLUMN] >= 0).all()


def test_tree_encoder_keeps_missing_numbers() -> None:
    train = make_frame()
    test = make_frame(3, seed=1)
    test.loc[0, config.MOMENTUM_COLUMN] = np.nan
    out = models.encode(models.fit_encoder(train), test)
    assert np.isnan(out.loc[0, config.MOMENTUM_COLUMN])
    pd.testing.assert_series_equal(
        out[config.VOLATILITY_COLUMN],
        test[config.VOLATILITY_COLUMN].astype("float64"),
    )


def test_linear_encoder_scales_with_training_stats_and_ignores_unseen() -> None:
    train = make_frame()
    encoder = models.fit_encoder(train, kind="linear")
    encoded_train = models.encode(encoder, train)
    numerical, _ = models.feature_columns(train)
    assert np.allclose(encoded_train[numerical].mean(), 0.0, atol=1e-9)

    test = make_frame(4, seed=1)
    test.loc[0, config.COMMODITY_COLUMN] = "Garlic"
    test.loc[1, config.MOMENTUM_COLUMN] = np.nan
    out = models.encode(encoder, test)
    onehot = [c for c in out.columns if c.startswith(config.COMMODITY_COLUMN + "_")]
    assert len(onehot) == len(COMMODITIES)
    assert out.loc[0, onehot].sum() == 0
    assert out.notna().all().all()


def test_fit_encoder_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        models.fit_encoder(make_frame(), kind="forest")


def test_encode_leaves_input_unchanged() -> None:
    train = make_frame()
    before = train.copy()
    models.encode(models.fit_encoder(train, kind="linear"), train)
    pd.testing.assert_frame_equal(train, before)


# Baselines


def test_majority_baseline_predicts_most_frequent_label() -> None:
    train = make_frame()
    train.loc[train.index[:5], LABEL] = np.nan
    model = models.fit_majority_baseline(train)
    pred = models.predict_majority_baseline(model, make_frame(7, seed=1))
    expected = train[LABEL].value_counts().idxmax()
    assert (pred == expected).all()
    assert len(pred) == 7


def test_majority_baseline_needs_labelled_rows() -> None:
    train = make_frame(5)
    train[LABEL] = np.nan
    with pytest.raises(ValueError):
        models.fit_majority_baseline(train)


def test_current_ratio_and_persistence_use_training_thresholds() -> None:
    df = pd.DataFrame(
        {
            config.COMMODITY_COLUMN: ["Onion"] * 4,
            "Modal_Price": [50.0, 95.0, 120.0, 80.0],
            config.HISTORY_MEAN_COLUMN: [100.0, 100.0, 100.0, 0.0],
        }
    )
    with_ratio = models.add_current_ratio(df)
    ratios = with_ratio[config.CURRENT_RATIO_COLUMN]
    assert np.allclose(ratios[:3], [0.5, 0.95, 1.2])
    assert np.isnan(ratios[3])

    thresholds = pd.DataFrame(
        {
            config.COMMODITY_COLUMN: ["Onion"],
            config.HIGH_THRESHOLD_COLUMN: [0.8],
            config.MEDIUM_THRESHOLD_COLUMN: [1.0],
        }
    )
    pred = models.predict_risk_persistence(df, thresholds)
    assert pred[:3].tolist() == ["High", "Medium", "Low"]
    assert pd.isna(pred[3])


def test_persistence_with_fitted_thresholds_and_missing_history() -> None:
    train = pd.DataFrame(
        {
            config.COMMODITY_COLUMN: ["Onion"] * 10,
            config.RATIO_COLUMN: np.linspace(0.5, 1.4, 10),
        }
    )
    thresholds = fit_risk_thresholds(train)
    df = pd.DataFrame(
        {
            config.COMMODITY_COLUMN: ["Onion", "Garlic"],
            "Modal_Price": [40.0, 40.0],
            config.HISTORY_MEAN_COLUMN: [100.0, 100.0],
        }
    )
    pred = models.predict_risk_persistence(df, thresholds)
    assert pred[0] == "High"
    assert pd.isna(pred[1])  # no thresholds for an unseen commodity
    with pytest.raises(KeyError):
        models.add_current_ratio(df.drop(columns=config.HISTORY_MEAN_COLUMN))


def test_logistic_baseline_fits_and_predicts() -> None:
    train = make_frame(120)
    model = models.fit_logistic_baseline(train)
    assert isinstance(model, Pipeline)
    test = make_frame(20, seed=1)
    pred = models.predict_logistic_baseline(model, test)
    assert_valid_predictions(pred, test)
    assert (pred == test[LABEL]).mean() > 0.5


def test_logistic_baseline_skips_unlabelled_rows() -> None:
    train = make_frame(120)
    train.loc[train.index[:10], LABEL] = np.nan
    model = models.fit_logistic_baseline(train)
    assert set(model.classes_) <= set(config.RISK_LABELS)
    assert model.named_steps["classify"].n_features_in_ > 0


# Main models


def test_random_forest_fits_and_predicts() -> None:
    train = make_frame(150)
    model = models.fit_random_forest(train, n_estimators=20, min_samples_leaf=1)
    forest = model.named_steps["classify"]
    assert forest.class_weight == config.CLASS_WEIGHT
    test = make_frame(30, seed=1)
    test.loc[0, config.COMMODITY_COLUMN] = "Garlic"
    pred = models.predict_random_forest(model, test)
    assert_valid_predictions(pred, test)
    assert (pred == test[LABEL]).mean() > 0.5


def test_random_forest_is_reproducible() -> None:
    train = make_frame(80)
    test = make_frame(20, seed=1)
    a = models.predict_random_forest(models.fit_random_forest(train, n_estimators=10), test)
    b = models.predict_random_forest(models.fit_random_forest(train, n_estimators=10), test)
    pd.testing.assert_series_equal(a, b)


def test_xgboost_fits_and_predicts_without_validation() -> None:
    train = make_frame(150)
    model = models.fit_xgboost(train, n_estimators=20, max_depth=3)
    assert model.named_steps["classify"].n_estimators == 20
    test = make_frame(30, seed=1)
    pred = models.predict_xgboost(model, test)
    assert_valid_predictions(pred, test)
    assert (pred == test[LABEL]).mean() > 0.5


def test_xgboost_stops_early_on_validation_frame() -> None:
    train = make_frame(150)
    valid = make_frame(40, seed=2)
    model = models.fit_xgboost(
        train, valid, n_estimators=300, learning_rate=0.3, early_stopping_rounds=5
    )
    booster = model.named_steps["classify"]
    assert booster.best_iteration < 299
    pred = models.predict_xgboost(model, make_frame(10, seed=3))
    assert pred.notna().all()


def test_xgboost_needs_every_risk_class() -> None:
    train = make_frame(60)
    train = train[train[LABEL] != "High"]
    with pytest.raises(ValueError):
        models.fit_xgboost(train, n_estimators=5)
