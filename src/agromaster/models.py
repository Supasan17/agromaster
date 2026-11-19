"""Model inputs, encoders and classifiers for the risk label.

Follows the proposal's "Feature engineering" and "Model development"
subsections. Learned preprocessing is fitted on the training part of a fold
only and then applied unchanged to validation or test data:

* tree models use ordinal codes for the categorical features, with unseen
  and missing categories mapped to ``config.UNKNOWN_CATEGORY_CODE``, and the
  numerical features passed through unchanged;
* the logistic regression baseline uses one-hot encoding for the categorical
  features (unseen categories become all zeros) and standardised numerical
  features, with missing values imputed by the training median.

Three baselines give reference points: a majority-class predictor, a
risk-persistence predictor that applies the training-fitted label thresholds
to the current price-to-history ratio, and a regularised multinomial logistic
regression on the linear encoder.

The first main model is a Random Forest on the tree encoder, with balanced
class weights fitted on the training rows only. The second main model is
an XGBoost classifier on the same encoder, trained with balanced class
weights as sample weights and, when a validation frame is given, early
stopping on the validation loss.

Encoders are scikit-learn ``ColumnTransformer`` objects whose output is a
pandas DataFrame; :func:`encode` applies a fitted encoder and leaves its input
unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from agromaster import config
from agromaster.features import predictor_price_columns
from agromaster.labels import apply_risk_thresholds

ENCODER_KINDS = ("tree", "linear")


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the ``(numerical, categorical)`` model input columns for ``df``.

    The lists follow ``config.NUMERICAL_FEATURES`` and
    ``config.CATEGORICAL_FEATURES``. The price features are replaced by the
    forward-filled predictor copies when ``df`` has them, so the models never
    see unfilled target-side prices from aligned data.
    """
    prices = dict(zip(config.PRICE_FEATURES, predictor_price_columns(df)))
    numerical = [prices.get(c, c) for c in config.NUMERICAL_FEATURES]
    return numerical, list(config.CATEGORICAL_FEATURES)


def _to_float(x: pd.DataFrame) -> pd.DataFrame:
    """Cast numerical inputs (including nullable integers) to float64 with NaN."""
    return x.astype("float64")


def _to_object(x: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical inputs to plain objects with missing values as NaN."""
    return x.astype("object").where(x.notna(), np.nan)


def _float_step() -> FunctionTransformer:
    return FunctionTransformer(_to_float, feature_names_out="one-to-one")


def _object_step() -> FunctionTransformer:
    return FunctionTransformer(_to_object, feature_names_out="one-to-one")


def make_tree_encoder(
    numerical: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Return an unfitted encoder for the tree models.

    Categorical columns get training-fitted ordinal codes; a category not
    seen during fitting, or a missing one, is coded
    ``config.UNKNOWN_CATEGORY_CODE``. Numerical columns are cast to float
    and passed through, keeping missing values for the trees to handle.
    """
    categories = Pipeline(
        [
            ("object", _object_step()),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=config.UNKNOWN_CATEGORY_CODE,
                    encoded_missing_value=config.UNKNOWN_CATEGORY_CODE,
                    dtype=np.float64,
                ),
            ),
        ]
    )
    encoder = ColumnTransformer(
        [("num", _float_step(), numerical), ("cat", categories, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def make_linear_encoder(
    numerical: list[str], categorical: list[str]
) -> ColumnTransformer:
    """Return an unfitted encoder for the logistic regression baseline.

    Numerical columns are imputed with the training median and
    standardised with the training mean and standard deviation. Categorical
    columns are one-hot encoded; missing values form their own category
    (``config.MISSING_CATEGORY``) and categories not seen during fitting
    are encoded as all zeros.
    """
    numbers = Pipeline(
        [
            ("float", _float_step()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categories = Pipeline(
        [
            ("object", _object_step()),
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value=config.MISSING_CATEGORY
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    encoder = ColumnTransformer(
        [("num", numbers, numerical), ("cat", categories, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def fit_encoder(train: pd.DataFrame, kind: str = "tree") -> ColumnTransformer:
    """Fit a ``"tree"`` or ``"linear"`` encoder on a training frame.

    The input columns come from :func:`feature_columns`. The returned
    encoder holds only training-fitted parameters and is applied to other
    frames with :func:`encode`.
    """
    if kind not in ENCODER_KINDS:
        raise ValueError(f"kind must be one of {ENCODER_KINDS}, got {kind!r}")
    numerical, categorical = feature_columns(train)
    make = make_tree_encoder if kind == "tree" else make_linear_encoder
    return make(numerical, categorical).fit(train)


def encode(encoder: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted encoder and return the model matrix as a DataFrame.

    The result keeps the index of ``df`` and has one column per encoded
    feature in the order fixed at fitting time.
    """
    encoded = encoder.transform(df)
    encoded.index = df.index
    return encoded


def _as_labels(values: np.ndarray, index: pd.Index) -> pd.Series:
    """Wrap predicted labels as an ordered categorical over the risk labels."""
    return pd.Series(
        pd.Categorical(values, categories=config.RISK_LABELS, ordered=True),
        index=index,
        name=config.LABEL_COLUMN,
    )


def fit_majority_baseline(
    train: pd.DataFrame, label_col: str = config.LABEL_COLUMN
) -> DummyClassifier:
    """Fit the majority-class baseline on the labelled rows of ``train``.

    The model always predicts the most frequent label among the training
    rows whose label is not missing. Raises ``ValueError`` when no row is
    labelled.
    """
    labels = train[label_col].dropna().astype("object")
    if labels.empty:
        raise ValueError("Cannot fit the majority baseline without labelled rows")
    model = DummyClassifier(strategy="most_frequent")
    return model.fit(np.zeros((len(labels), 1)), labels.to_numpy())


def predict_majority_baseline(
    model: DummyClassifier, df: pd.DataFrame
) -> pd.Series:
    """Predict the majority-class label for every row of ``df``."""
    values = model.predict(np.zeros((len(df), 1)))
    return _as_labels(values, df.index)


def add_current_ratio(
    df: pd.DataFrame,
    price_col: str | None = None,
    history_col: str = config.HISTORY_MEAN_COLUMN,
) -> pd.DataFrame:
    """Add the current price-to-history ratio used by the persistence baseline.

    ``config.CURRENT_RATIO_COLUMN`` is the modal price on the prediction date
    divided by ``history_col`` (the mean modal price over the
    ``config.HISTORY_DAYS`` days before that date, as added by
    :func:`agromaster.labels.add_future_ratio`). ``price_col`` defaults to
    the modal predictor price (the forward-filled copy when present), so the
    ratio uses only information available on the prediction date. It is
    missing where either input is missing or the history mean is not
    positive.
    """
    if history_col not in df.columns:
        raise KeyError(
            f"Column {history_col!r} is missing; add it with "
            "agromaster.labels.add_future_ratio first"
        )
    modal_col = price_col or predictor_price_columns(df)[2]
    out = df.copy()
    price = out[modal_col].to_numpy(dtype="float64")
    history = out[history_col].to_numpy(dtype="float64")
    safe_history = np.where(history > 0, history, np.nan)
    out[config.CURRENT_RATIO_COLUMN] = (price / safe_history).astype("float32")
    return out


def predict_risk_persistence(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    price_col: str | None = None,
    history_col: str = config.HISTORY_MEAN_COLUMN,
) -> pd.Series:
    """Predict the risk label seven days ahead by persisting the current risk.

    The current ratio from :func:`add_current_ratio` is labelled with the
    training-fitted commodity ``thresholds`` (from
    :func:`agromaster.labels.fit_risk_thresholds`, used unchanged), and that
    class is the prediction. The prediction is missing where the current
    ratio is missing or the commodity has no thresholds.
    """
    current = add_current_ratio(df, price_col, history_col)
    labelled = apply_risk_thresholds(
        current, thresholds, ratio_col=config.CURRENT_RATIO_COLUMN
    )
    return labelled[config.LABEL_COLUMN]


def _labelled(
    train: pd.DataFrame, label_col: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the labelled rows of ``train`` and their labels as objects."""
    mask = train[label_col].notna().to_numpy()
    if not mask.any():
        raise ValueError("Cannot fit a model without labelled rows")
    rows = train[mask]
    return rows, rows[label_col].astype("object").to_numpy()


def fit_logistic_baseline(
    train: pd.DataFrame,
    label_col: str = config.LABEL_COLUMN,
    c: float = config.LOGREG_C,
    class_weight: str | None = config.CLASS_WEIGHT,
) -> Pipeline:
    """Fit the regularised multinomial logistic regression baseline.

    The model is a pipeline of the linear encoder (one-hot categories,
    median-imputed and standardised numbers) and an L2-regularised
    multinomial ``LogisticRegression`` with inverse strength ``c``. Both
    steps, and the class weights, are fitted on the labelled rows of
    ``train`` only, so the pipeline can be applied unchanged to another
    frame with :func:`predict_logistic_baseline`.
    """
    rows, labels = _labelled(train, label_col)
    numerical, categorical = feature_columns(rows)
    model = Pipeline(
        [
            ("encode", make_linear_encoder(numerical, categorical)),
            (
                "classify",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    max_iter=config.LOGREG_MAX_ITER,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    return model.fit(rows, labels)


def predict_logistic_baseline(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict the risk label for every row of ``df`` with a fitted baseline."""
    return _as_labels(model.predict(df), df.index)


def fit_random_forest(
    train: pd.DataFrame,
    label_col: str = config.LABEL_COLUMN,
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int | None = config.RF_MAX_DEPTH,
    min_samples_leaf: int = config.RF_MIN_SAMPLES_LEAF,
    class_weight: str | None = config.CLASS_WEIGHT,
) -> Pipeline:
    """Fit the Random Forest risk classifier.

    The model is a pipeline of the tree encoder (training-fitted ordinal
    codes for the categories, numerical features passed through with their
    missing values) and a ``RandomForestClassifier`` with ``n_estimators``
    trees, the given ``max_depth`` and ``min_samples_leaf``, and balanced
    class weights by default. Everything is fitted on the labelled rows of
    ``train`` only, so the pipeline can be applied unchanged to another
    frame with :func:`predict_random_forest`.
    """
    rows, labels = _labelled(train, label_col)
    numerical, categorical = feature_columns(rows)
    model = Pipeline(
        [
            ("encode", make_tree_encoder(numerical, categorical)),
            (
                "classify",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    n_jobs=config.RF_N_JOBS,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    return model.fit(rows, labels)


def predict_random_forest(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict the risk label for every row of ``df`` with a fitted forest."""
    return _as_labels(model.predict(df), df.index)


def _label_codes(labels: np.ndarray) -> np.ndarray:
    """Map risk labels to the integer codes XGBoost needs (their position)."""
    codes = pd.Categorical(labels, categories=config.RISK_LABELS).codes
    if (codes < 0).any():
        raise ValueError(f"Labels must be in {config.RISK_LABELS}")
    return codes.astype(np.int64)


def _class_weights(
    codes: np.ndarray, class_weight: str | None
) -> dict[int, float]:
    """Return the per-class weight for each label code seen in training."""
    present = np.unique(codes)
    if class_weight is None:
        return {int(c): 1.0 for c in present}
    weights = compute_class_weight(class_weight, classes=present, y=codes)
    return {int(c): float(w) for c, w in zip(present, weights)}


def fit_xgboost(
    train: pd.DataFrame,
    valid: pd.DataFrame | None = None,
    label_col: str = config.LABEL_COLUMN,
    n_estimators: int = config.XGB_N_ESTIMATORS,
    learning_rate: float = config.XGB_LEARNING_RATE,
    max_depth: int = config.XGB_MAX_DEPTH,
    reg_lambda: float = config.XGB_REG_LAMBDA,
    early_stopping_rounds: int | None = config.XGB_EARLY_STOPPING_ROUNDS,
    class_weight: str | None = config.CLASS_WEIGHT,
) -> Pipeline:
    """Fit the XGBoost risk classifier.

    The model is a pipeline of the tree encoder and an ``XGBClassifier``
    with up to ``n_estimators`` trees, the given ``learning_rate``,
    ``max_depth`` and L2 regularisation ``reg_lambda``. The encoder and the
    class weights are fitted on the labelled rows of ``train`` only; each
    training row is weighted by its class weight (balanced by default).
    Every risk class must occur in the training labels.

    When ``valid`` is given, its labelled rows are encoded with the
    training-fitted encoder, weighted with the training class weights and
    used for early stopping: boosting stops once the validation loss has not
    improved for ``early_stopping_rounds`` rounds, and the best iteration is
    used for prediction. Without ``valid`` all ``n_estimators`` trees are
    fitted. Apply the pipeline to another frame with
    :func:`predict_xgboost`.
    """
    rows, labels = _labelled(train, label_col)
    codes = _label_codes(labels)
    missing = set(range(len(config.RISK_LABELS))) - set(np.unique(codes))
    if missing:
        absent = [config.RISK_LABELS[c] for c in sorted(missing)]
        raise ValueError(f"Training labels have no rows for {absent}")
    weights = _class_weights(codes, class_weight)

    numerical, categorical = feature_columns(rows)
    encoder = make_tree_encoder(numerical, categorical).fit(rows)
    x_train = encode(encoder, rows)
    fit_kwargs: dict = {
        "sample_weight": np.array([weights[c] for c in codes]),
        "verbose": False,
    }
    stopping = None
    if valid is not None:
        valid_rows, valid_labels = _labelled(valid, label_col)
        valid_codes = _label_codes(valid_labels)
        fit_kwargs["eval_set"] = [(encode(encoder, valid_rows), valid_codes)]
        fit_kwargs["sample_weight_eval_set"] = [
            np.array([weights.get(c, 1.0) for c in valid_codes])
        ]
        stopping = early_stopping_rounds

    classifier = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        reg_lambda=reg_lambda,
        objective="multi:softprob",
        eval_metric=config.XGB_EVAL_METRIC,
        early_stopping_rounds=stopping,
        tree_method=config.XGB_TREE_METHOD,
        n_jobs=config.XGB_N_JOBS,
        random_state=config.SEED,
    )
    classifier.fit(x_train, codes, **fit_kwargs)
    return Pipeline([("encode", encoder), ("classify", classifier)])


def predict_xgboost(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Predict the risk label for every row of ``df`` with a fitted XGBoost."""
    codes = model.predict(df)
    labels = np.asarray(config.RISK_LABELS, dtype=object)[codes]
    return _as_labels(labels, df.index)
