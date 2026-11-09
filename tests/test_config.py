"""Tests for the project constants and paths in agromaster.config."""
from pathlib import Path

import pandas as pd
import pytest

from agromaster import config


@pytest.mark.parametrize(
    "path",
    [
        config.DATA_RAW,
        config.DATA_INTERIM,
        config.DATA_PROCESSED,
        config.MODELS,
        config.FIGURES,
    ],
)
def test_project_paths_exist(path: Path) -> None:
    assert path.is_dir()
    assert config.ROOT in path.parents


def test_root_is_project_root() -> None:
    assert (config.ROOT / "pyproject.toml").is_file()


def test_target_constants_match_proposal() -> None:
    assert config.HORIZON_DAYS == 7
    assert config.HISTORY_DAYS == 90
    assert config.HIGH_RISK_QUANTILE == 0.10
    assert config.MEDIUM_RISK_QUANTILE == 0.30
    assert 0 < config.HIGH_RISK_QUANTILE < config.MEDIUM_RISK_QUANTILE < 1


def test_preprocessing_and_cv_constants() -> None:
    assert config.MAX_FFILL_DAYS == 3
    assert config.N_CV_FOLDS == 5
    assert isinstance(config.SEED, int)


def test_series_keys_are_unique() -> None:
    assert len(config.SERIES_KEYS) == len(set(config.SERIES_KEYS))
    assert "Commodity" in config.SERIES_KEYS
    assert "Market" in config.SERIES_KEYS


def test_split_dates_are_ordered() -> None:
    assert pd.Timestamp(config.DEV_END) < pd.Timestamp(config.TEST_START)
