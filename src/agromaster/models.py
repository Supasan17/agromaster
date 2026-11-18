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

Encoders are scikit-learn ``ColumnTransformer`` objects whose output is a
pandas DataFrame; :func:`encode` applies a fitted encoder and leaves its input
unchanged.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from agromaster import config
from agromaster.features import predictor_price_columns

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
