"""Tests for the chronological split and CV folds in agromaster.split."""
import numpy as np
import pandas as pd
import pytest

from agromaster import config, split

DATE = config.DATE_COLUMN


def make_frame(start: str, days: int, per_day: int = 2) -> pd.DataFrame:
    """Frame with ``per_day`` records on each of ``days`` consecutive dates."""
    dates = pd.date_range(start, periods=days, freq="D").repeat(per_day)
    return pd.DataFrame(
        {
            "Market": np.tile([f"M{i}" for i in range(per_day)], days),
            DATE: dates,
            "Value": np.arange(len(dates), dtype="float64"),
        }
    )


# Target purge


def test_purge_keeps_only_rows_whose_target_is_before_boundary() -> None:
    df = make_frame("2024-01-01", 10, per_day=1)
    out = split.purge_target_overlap(df, "2024-01-10", horizon_days=3)
    # Target dates t + 3 must be before Jan 10, so t <= Jan 6.
    assert out[DATE].max() == pd.Timestamp("2024-01-06")
    assert len(out) == 6


def test_purge_with_zero_horizon_cuts_at_boundary() -> None:
    df = make_frame("2024-01-01", 5, per_day=1)
    out = split.purge_target_overlap(df, "2024-01-04", horizon_days=0)
    assert out[DATE].tolist() == list(pd.date_range("2024-01-01", periods=3))


def test_purge_drops_missing_dates_and_rejects_negative_horizon() -> None:
    df = make_frame("2024-01-01", 3, per_day=1)
    df[DATE] = df[DATE].astype("object")
    df.loc[1, DATE] = None
    out = split.purge_target_overlap(df, "2024-02-01", horizon_days=1)
    assert len(out) == 2
    with pytest.raises(ValueError):
        split.purge_target_overlap(df, "2024-02-01", horizon_days=-1)


def test_purge_ignores_time_of_day() -> None:
    df = pd.DataFrame({DATE: pd.to_datetime(["2024-01-02 23:00", "2024-01-03 01:00"])})
    out = split.purge_target_overlap(df, "2024-01-10 12:00", horizon_days=7)
    # Jan 2 + 7 = Jan 9 is before Jan 10; Jan 3 + 7 = Jan 10 is not.
    assert len(out) == 1


# Chronological dev/test split


def test_split_partitions_by_date_and_purges_dev_tail() -> None:
    df = make_frame("2023-12-01", 62)  # Dec 1 2023 to Jan 31 2024
    dev, test = split.chronological_split(
        df, dev_end="2023-12-31", test_start="2024-01-01", horizon_days=7
    )
    assert dev[DATE].max() == pd.Timestamp("2023-12-24")
    assert test[DATE].min() == pd.Timestamp("2024-01-01")
    assert len(test) == 31 * 2
    assert len(dev) == 24 * 2


def test_no_dev_target_reaches_test_period() -> None:
    df = make_frame("2023-10-01", 120, per_day=3)
    dev, test = split.chronological_split(
        df, dev_end="2023-12-31", test_start="2024-01-01"
    )
    target = dev[DATE] + pd.Timedelta(days=config.HORIZON_DAYS)
    assert (target < test[DATE].min()).all()
    assert set(dev.index).isdisjoint(test.index)


def test_split_keeps_records_of_a_date_together() -> None:
    df = make_frame("2023-12-20", 20, per_day=4).sample(frac=1.0, random_state=0)
    dev, test = split.chronological_split(
        df, dev_end="2023-12-31", test_start="2024-01-01", horizon_days=2
    )
    for part in (dev, test):
        counts = part.groupby(DATE).size()
        assert (counts == 4).all()


def test_gap_between_dev_end_and_test_start_is_in_neither() -> None:
    df = make_frame("2023-12-01", 40, per_day=1)
    dev, test = split.chronological_split(
        df, dev_end="2023-12-10", test_start="2023-12-20", horizon_days=0
    )
    assert dev[DATE].max() == pd.Timestamp("2023-12-10")
    assert test[DATE].min() == pd.Timestamp("2023-12-20")
    assert len(dev) + len(test) == 10 + 21


def test_split_preserves_order_index_and_input() -> None:
    df = make_frame("2023-12-25", 14).sample(frac=1.0, random_state=1)
    before = df.copy()
    dev, test = split.chronological_split(
        df, dev_end="2023-12-31", test_start="2024-01-01", horizon_days=1
    )
    pd.testing.assert_frame_equal(df, before)
    for part in (dev, test):
        positions = df.index.get_indexer(part.index)
        assert (np.diff(positions) > 0).all()


def test_split_rejects_dev_end_not_before_test_start() -> None:
    df = make_frame("2024-01-01", 5)
    with pytest.raises(ValueError):
        split.chronological_split(df, dev_end="2024-01-03", test_start="2024-01-03")


# Time blocks


def test_time_blocks_cover_distinct_dates_evenly() -> None:
    df = make_frame("2024-01-01", 12, per_day=3)
    starts = split.time_block_starts(df, n_blocks=6)
    assert starts == list(pd.date_range("2024-01-01", periods=6, freq="2D"))


def test_time_blocks_default_to_config_fold_count() -> None:
    df = make_frame("2024-01-01", 60)
    assert len(split.time_block_starts(df)) == config.N_CV_FOLDS + 1


def test_time_blocks_reject_too_few_dates_or_blocks() -> None:
    df = make_frame("2024-01-01", 3)
    with pytest.raises(ValueError):
        split.time_block_starts(df, n_blocks=4)
    with pytest.raises(ValueError):
        split.time_block_starts(df, n_blocks=1)


# Expanding-window CV folds


def test_folds_expand_and_validate_on_next_block() -> None:
    df = make_frame("2024-01-01", 60, per_day=2)
    folds = split.expanding_window_folds(df, n_folds=5, horizon_days=0)
    assert len(folds) == 5
    dates = df[DATE]
    starts = split.time_block_starts(df, n_blocks=6)
    for k, (train, val) in enumerate(folds, start=1):
        assert dates.iloc[train].max() < starts[k]
        assert dates.iloc[val].min() == starts[k]
        if k < 5:
            assert dates.iloc[val].max() < starts[k + 1]
        else:
            assert dates.iloc[val].max() == dates.max()
    sizes = [len(train) for train, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == 5


def test_fold_training_targets_never_reach_validation() -> None:
    df = make_frame("2024-01-01", 120, per_day=3)
    folds = split.expanding_window_folds(df, n_folds=5, horizon_days=7)
    dates = df[DATE]
    for train, val in folds:
        target = dates.iloc[train] + pd.Timedelta(days=7)
        assert (target < dates.iloc[val].min()).all()
        assert np.intersect1d(train, val).size == 0


def test_fold_purge_matches_purge_function() -> None:
    df = make_frame("2024-01-01", 60, per_day=2)
    folds = split.expanding_window_folds(df, n_folds=3, horizon_days=5)
    starts = split.time_block_starts(df, n_blocks=4)
    for k, (train, _) in enumerate(folds, start=1):
        earlier = df[df[DATE] < starts[k]]
        expected = split.purge_target_overlap(earlier, starts[k], horizon_days=5)
        assert df.index[train].tolist() == expected.index.tolist()


def test_folds_keep_dates_whole_and_return_sorted_positions() -> None:
    df = make_frame("2024-01-01", 30, per_day=3).sample(frac=1.0, random_state=2)
    df = df.reset_index(drop=True)
    folds = split.expanding_window_folds(df, n_folds=4, horizon_days=1)
    for train, val in folds:
        for idx in (train, val):
            assert (np.diff(idx) > 0).all()
            counts = df.iloc[idx].groupby(DATE).size()
            assert (counts == 3).all()


def test_folds_skip_rows_without_date_and_reject_negative_horizon() -> None:
    df = make_frame("2024-01-01", 12, per_day=1)
    df[DATE] = df[DATE].astype("object")
    df.loc[5, DATE] = None
    folds = split.expanding_window_folds(df, n_folds=2, horizon_days=0)
    for train, val in folds:
        assert 5 not in train and 5 not in val
    with pytest.raises(ValueError):
        split.expanding_window_folds(df, n_folds=2, horizon_days=-1)
