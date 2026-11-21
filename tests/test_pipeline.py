"""End-to-end test of the pipeline stages on synthetic data.

A small raw-style CSV is written to ``tmp_path`` and run through every stage
after ``download``: conversion to Parquet, cleaning, features, training and
evaluation, both by calling the stage functions and through the command-line
entry point. The real dataset is never needed. The classifiers are fitted
with tiny sizes to keep the test fast; everything else uses the defaults.
"""
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agromaster import __main__ as cli
from agromaster import config, models, pipeline
from agromaster.convert import convert_csv_to_parquet

RAW_HEADER = [
    "State", "District", "Market", "Commodity", "Variety", "Grade",
    "Arrival_Date", "Min Price", "Max Price", "Modal Price",
]
SERIES = [
    ("Kerala", "Ernakulam", "Aluva", "Banana", "Nendra", "FAQ", 3000.0),
    ("Kerala", "Ernakulam", "Aluva", "Onion", "Red", "FAQ", 1800.0),
    ("Punjab", "Ludhiana", "Khanna", "Banana", "Other", "FAQ", 2800.0),
    ("Punjab", "Ludhiana", "Khanna", "Onion", "Other", "FAQ", 1500.0),
]
START = "2023-01-01"
END = "2024-06-30"
# A test period that starts after the development period in config, so the
# chronological split leaves rows on both sides.
assert pd.Timestamp(config.TEST_START) < pd.Timestamp(END)


def write_raw_csv(path: Path, seed: int = 0) -> Path:
    """Write daily random-walk prices for a few series, with gaps and bad rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    frames = []
    for *keys, base in SERIES:
        # Keep roughly 85% of the days, so calendar alignment has gaps to fill.
        days = dates[rng.random(len(dates)) < 0.85]
        modal = base * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(days))))
        spread = rng.uniform(0.02, 0.10, len(days)) * modal
        frame = pd.DataFrame(
            {
                **dict(zip(RAW_HEADER[:6], keys, strict=True)),
                "Arrival_Date": days.strftime("%d/%m/%Y"),
                "Min Price": (modal - spread).round(),
                "Max Price": (modal + spread).round(),
                "Modal Price": modal.round(),
            }
        )
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    # Rows the cleaning stage must drop: an exact duplicate, a missing price
    # and a non-positive price.
    bad = raw.iloc[[0, 1, 2]].copy()
    bad.iloc[1, bad.columns.get_loc("Modal Price")] = np.nan
    bad.iloc[2, bad.columns.get_loc("Modal Price")] = 0.0
    raw = pd.concat([raw, bad], ignore_index=True)
    raw.to_csv(path, index=False, columns=RAW_HEADER)
    return path


@pytest.fixture(scope="module", autouse=True)
def small_models() -> Iterator[None]:
    """Fit the tree ensembles with a few small trees for the whole module."""
    small = {
        "random_forest": partial(models.fit_random_forest, n_estimators=10, max_depth=4),
        "xgboost": partial(models.fit_xgboost, n_estimators=10, max_depth=3),
    }
    with pytest.MonkeyPatch.context() as mp:
        for name, fit in small.items():
            _, predict = pipeline.FITTED_MODELS[name]
            mp.setitem(pipeline.FITTED_MODELS, name, (fit, predict))
        yield


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run convert, clean, features, train and evaluate once in a temp folder."""
    root = tmp_path_factory.mktemp("pipeline")
    paths = {
        "raw": write_raw_csv(root / "raw.csv"),
        "interim": root / "interim" / "prices.parquet",
        "clean": root / "processed" / "clean.parquet",
        "features": root / "processed" / "features.parquet",
        "models": root / "models",
        "reports": root / "reports",
    }
    convert_csv_to_parquet(paths["raw"], paths["interim"], chunk_rows=500)
    pipeline.run_clean(paths["interim"], paths["clean"])
    pipeline.run_features(paths["clean"], paths["features"])
    pipeline.run_train(paths["features"], paths["models"])
    pipeline.run_evaluate(paths["features"], paths["models"], paths["reports"])
    return paths


def test_stages_write_their_outputs(run: dict[str, Path]) -> None:
    for stage in ["interim", "clean", "features"]:
        assert run[stage].exists()
    assert (run["models"] / config.THRESHOLDS_FILE.name).exists()
    for name in pipeline.FITTED_MODELS:
        assert pipeline.model_path(name, run["models"]).exists()
    assert (run["reports"] / "metrics_summary.csv").exists()


def test_cleaning_drops_bad_rows_and_aligns_series(run: dict[str, Path]) -> None:
    clean = pd.read_parquet(run["clean"])
    observed = clean[clean[config.OBSERVED_COLUMN]]
    assert (observed["Modal_Price"] > 0).all()
    assert not observed.duplicated(
        [*config.SERIES_KEYS, config.DATE_COLUMN]
    ).any()
    # Every series is aligned to one row per calendar day.
    days = clean.groupby(config.SERIES_KEYS, observed=True)[config.DATE_COLUMN].agg(
        lambda d: d.diff().dropna().dt.days.eq(1).all()
    )
    assert len(days) == len(SERIES)
    assert days.all()


def test_features_include_model_inputs_and_ratio(run: dict[str, Path]) -> None:
    features = pd.read_parquet(run["features"])
    for column in [*config.NUMERICAL_FEATURES, *config.CATEGORICAL_FEATURES]:
        assert column in features.columns
    assert config.RATIO_COLUMN in features.columns
    assert features[config.RATIO_COLUMN].notna().any()


def test_thresholds_are_fitted_on_development_period_only(
    run: dict[str, Path],
) -> None:
    saved = pd.read_parquet(run["models"] / config.THRESHOLDS_FILE.name)
    features = pd.read_parquet(run["features"])
    _, _, refitted = pipeline.labelled_split(features)
    pd.testing.assert_frame_equal(
        saved.reset_index(drop=True), refitted.reset_index(drop=True)
    )
    high = saved[config.HIGH_THRESHOLD_COLUMN]
    assert (high <= saved[config.MEDIUM_THRESHOLD_COLUMN]).all()


def test_evaluation_scores_every_model_on_test_period(run: dict[str, Path]) -> None:
    summary = pd.read_csv(run["reports"] / "metrics_summary.csv", index_col="Model")
    assert list(summary.index) == config.MODEL_NAMES
    scores = summary.drop(columns="N")
    assert not scores.empty
    assert ((scores >= 0) & (scores <= 1)).all().all()
    # Every model is scored on the same labelled test rows.
    assert summary["N"].nunique() == 1
    assert summary["N"].iloc[0] > 0
    for name in config.MODEL_NAMES:
        for report in ["per_class", "confusion", "per_commodity", "per_period"]:
            assert (run["reports"] / f"{name}_{report}.csv").exists()
    per_period = pd.read_csv(run["reports"] / "xgboost_per_period.csv")
    # The test period is only 2024 in the synthetic data.
    assert per_period.astype(str).apply(lambda c: c.str.contains("2024")).any().any()


def test_cli_runs_stages_with_configured_paths(
    run: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "INTERIM_PARQUET", run["interim"])
    monkeypatch.setattr(config, "CLEAN_PARQUET", tmp_path / "clean.parquet")
    monkeypatch.setattr(config, "FEATURES_PARQUET", tmp_path / "features.parquet")
    monkeypatch.setattr(config, "MODELS", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS", tmp_path / "reports")

    code = cli.main(["clean", "features", "train", "evaluate"])

    assert code == 0
    assert (tmp_path / "reports" / "metrics_summary.csv").exists()
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "features.parquet"),
        pd.read_parquet(run["features"]),
    )


def test_cli_reports_missing_stage_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(config, "INTERIM_PARQUET", tmp_path / "missing.parquet")

    code = cli.main(["clean"])

    assert code == 1
    assert "'convert' stage first" in capsys.readouterr().err


def test_cli_expands_all_into_every_stage() -> None:
    assert cli.resolve_stages(["all"]) == config.PIPELINE_STAGES
    assert cli.resolve_stages(["train", "clean"]) == ["train", "clean"]
