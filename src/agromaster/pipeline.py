"""Pipeline stages run by the command-line entry point.

Each stage reads the output of the stage before it from disk and writes its
own output, so the full pipeline can be rerun from raw data to final results
as the proposal's "Overall approach" asks:

1. ``download``: the Kaggle CSV into ``data/raw/``;
2. ``convert``: the CSV to Parquet in ``data/interim/``, chunk by chunk;
3. ``clean``: record filters, name standardisation, same-day aggregation and
   calendar alignment, written to ``config.CLEAN_PARQUET``;
4. ``features``: the predictors and the seven-day ahead price ratio, written
   to ``config.FEATURES_PARQUET``;
5. ``train``: the chronological development/test split, commodity risk
   thresholds fitted on the development period only, and the baselines and
   classifiers fitted on the labelled development rows, saved under
   ``config.MODELS``;
6. ``evaluate``: the saved models applied to the test period with the
   development thresholds unchanged, with overall, per-class, per-commodity
   and per-period metrics written to ``config.REPORTS``.

Every stage takes its input and output locations as arguments (defaulting to
the paths in ``config``) so it can be run on small synthetic data.
"""
from pathlib import Path

import joblib
import pandas as pd

from agromaster import config, evaluate, models
from agromaster.cleaning import (
    aggregate_same_day,
    align_calendar_days,
    filter_records,
    standardise_names,
)
from agromaster.convert import convert_csv_to_parquet
from agromaster.download import download_dataset
from agromaster.features import (
    add_calendar_features,
    add_lag_features,
    add_price_spread,
    add_rolling_features,
)
from agromaster.labels import (
    add_future_ratio,
    apply_risk_thresholds,
    fit_risk_thresholds,
)
from agromaster.split import chronological_split

# Fitted models: name -> (fit, predict). The risk-persistence baseline has no
# fitted model; it only needs the development thresholds.
FITTED_MODELS = {
    "majority": (models.fit_majority_baseline, models.predict_majority_baseline),
    "logistic": (models.fit_logistic_baseline, models.predict_logistic_baseline),
    "random_forest": (models.fit_random_forest, models.predict_random_forest),
    "xgboost": (models.fit_xgboost, models.predict_xgboost),
}


def _read(path: Path, stage: str) -> pd.DataFrame:
    """Read a stage input, or explain which stage has to run first."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the {stage!r} stage first.")
    return pd.read_parquet(path)


def _write(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to Parquet at ``path``, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def model_path(name: str, models_dir: Path | None = None) -> Path:
    """Return the file a fitted model called ``name`` is saved to."""
    return (models_dir or config.MODELS) / config.MODEL_FILE_TEMPLATE.format(name=name)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the preprocessing steps in the proposal's order."""
    return align_calendar_days(aggregate_same_day(standardise_names(filter_records(df))))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add the predictors and the label ratio to a cleaned, aligned frame."""
    out = add_price_spread(df)
    out = add_calendar_features(out)
    out = add_lag_features(out)
    out = add_rolling_features(out)
    return add_future_ratio(out)


def labelled_split(
    df: pd.DataFrame, thresholds: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically and label both parts.

    Thresholds are fitted on the development part unless given, and applied
    unchanged to both parts. Returns ``(dev, test, thresholds)``.
    """
    dev, test = chronological_split(df)
    if thresholds is None:
        thresholds = fit_risk_thresholds(dev)
    return (
        apply_risk_thresholds(dev, thresholds),
        apply_risk_thresholds(test, thresholds),
        thresholds,
    )


def run_download() -> Path:
    """Download the raw dataset (skipped when already present)."""
    return download_dataset()


def run_convert() -> int:
    """Convert the raw CSV to the interim Parquet; return rows written."""
    return convert_csv_to_parquet()


def run_clean(src: Path | None = None, dest: Path | None = None) -> Path:
    """Clean the interim table and write the aligned series."""
    df = _read(src or config.INTERIM_PARQUET, "convert")
    return _write(clean_frame(df), dest or config.CLEAN_PARQUET)


def run_features(src: Path | None = None, dest: Path | None = None) -> Path:
    """Add predictors and the label ratio to the cleaned series."""
    df = _read(src or config.CLEAN_PARQUET, "clean")
    return _write(feature_frame(df), dest or config.FEATURES_PARQUET)


def run_train(src: Path | None = None, models_dir: Path | None = None) -> list[Path]:
    """Fit the thresholds and models on the development period and save them."""
    models_dir = models_dir or config.MODELS
    dev, _, thresholds = labelled_split(
        _read(src or config.FEATURES_PARQUET, "features")
    )
    saved = [_write(thresholds, models_dir / config.THRESHOLDS_FILE.name)]
    for name, (fit, _) in FITTED_MODELS.items():
        path = model_path(name, models_dir)
        joblib.dump(fit(dev), path)
        saved.append(path)
    return saved


def predict_all(
    test: pd.DataFrame, thresholds: pd.DataFrame, models_dir: Path | None = None
) -> dict[str, pd.Series]:
    """Return the test predictions of every model in ``config.MODEL_NAMES``."""
    predictions = {}
    for name in config.MODEL_NAMES:
        if name == "persistence":
            pred = models.predict_risk_persistence(test, thresholds)
            # A missing current ratio gives no persistence prediction; fall
            # back to the class of no decline so every labelled row is scored.
            predictions[name] = pred.fillna(config.RISK_LABELS[0])
            continue
        _, predict = FITTED_MODELS[name]
        predictions[name] = predict(joblib.load(model_path(name, models_dir)), test)
    return predictions


def run_evaluate(
    src: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Score the saved models on the test period and write the reports.

    Writes ``metrics_summary.csv`` (one row per model) and, per model,
    per-class metrics, the confusion matrix and the per-commodity and
    per-period reports. Returns the tables keyed by file stem.
    """
    models_dir = models_dir or config.MODELS
    reports_dir = reports_dir or config.REPORTS
    thresholds = _read(models_dir / config.THRESHOLDS_FILE.name, "train")
    features = _read(src or config.FEATURES_PARQUET, "features")
    _, test, _ = labelled_split(features, thresholds)
    y_true = test[config.LABEL_COLUMN]

    tables: dict[str, pd.DataFrame] = {}
    summary = {}
    for name, y_pred in predict_all(test, thresholds, models_dir).items():
        summary[name] = evaluate.summary_metrics(y_true, y_pred)
        tables[f"{name}_per_class"] = evaluate.per_class_metrics(y_true, y_pred)
        tables[f"{name}_confusion"] = evaluate.confusion_matrix_frame(y_true, y_pred)
        tables[f"{name}_per_commodity"] = evaluate.per_commodity_report(test, y_pred)
        tables[f"{name}_per_period"] = evaluate.per_period_report(test, y_pred)
    tables["metrics_summary"] = pd.DataFrame(summary).T.rename_axis("Model")

    reports_dir.mkdir(parents=True, exist_ok=True)
    for stem, table in tables.items():
        table.to_csv(reports_dir / f"{stem}.csv")
    return tables


STAGES = {
    "download": run_download,
    "convert": run_convert,
    "clean": run_clean,
    "features": run_features,
    "train": run_train,
    "evaluate": run_evaluate,
}
