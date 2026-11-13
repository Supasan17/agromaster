"""Tests for the record filters and name standardisation in agromaster.cleaning."""
import numpy as np
import pandas as pd

from agromaster import cleaning, config


def make_records() -> pd.DataFrame:
    """Small interim-style frame with one duplicate and a few bad price rows."""
    return pd.DataFrame(
        {
            "Commodity": ["Wheat", "Wheat", "Rice", "Rice", "Onion", "Onion", "Potato"],
            config.DATE_COLUMN: pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "Min_Price": [2000.0, 2000.0, 3000.0, 0.0, 1000.0, 1200.0, 900.0],
            "Max_Price": [2400.0, 2400.0, 3500.0, 3400.0, 1500.0, 1100.0, 1300.0],
            "Modal_Price": [2200.0, 2200.0, np.nan, 3200.0, 1600.0, 1150.0, 1100.0],
        }
    )


def test_drop_duplicates_keeps_first() -> None:
    df = make_records()
    out = cleaning.drop_duplicates(df)
    assert len(out) == len(df) - 1
    assert list(out.index) == list(range(len(out)))
    assert len(df) == 7


def test_drop_invalid_prices_removes_missing_and_non_positive() -> None:
    out = cleaning.drop_invalid_prices(make_records())
    assert len(out) == 5
    assert out[config.PRICE_COLUMNS].notna().all().all()
    assert (out[config.PRICE_COLUMNS] > 0).all().all()


def test_drop_invalid_prices_respects_given_columns() -> None:
    out = cleaning.drop_invalid_prices(make_records(), price_cols=["Min_Price"])
    assert len(out) == 6
    assert out["Modal_Price"].isna().sum() == 1


def test_drop_inconsistent_prices_counts_violators() -> None:
    kept, n_dropped = cleaning.drop_inconsistent_prices(make_records())
    # Rice (missing modal), Onion modal > max, Onion min > modal are dropped;
    # the zero-min Rice row is ordered correctly and left for the price filter.
    assert n_dropped == 3
    assert len(kept) == 4
    assert (kept["Min_Price"] <= kept["Modal_Price"]).all()
    assert (kept["Modal_Price"] <= kept["Max_Price"]).all()


def test_drop_inconsistent_prices_keeps_equal_prices() -> None:
    df = pd.DataFrame({"Min_Price": [5.0], "Max_Price": [5.0], "Modal_Price": [5.0]})
    kept, n_dropped = cleaning.drop_inconsistent_prices(df)
    assert n_dropped == 0
    assert len(kept) == 1


def test_filter_records_applies_all_filters() -> None:
    df = make_records()
    out = cleaning.filter_records(df)
    assert out["Commodity"].tolist() == ["Wheat", "Potato"]
    assert len(df) == 7


def test_standardise_name_trims_and_title_cases() -> None:
    assert cleaning.standardise_name("  tamil   NADU ") == "Tamil Nadu"


def test_standardise_name_applies_alias() -> None:
    aliases = config.NAME_ALIASES["State"]
    assert cleaning.standardise_name(" ORISSA", aliases) == "Odisha"
    assert cleaning.standardise_name("Jammu  &  Kashmir", aliases) == "Jammu and Kashmir"


def test_standardise_names_on_object_and_categorical_columns() -> None:
    df = pd.DataFrame(
        {
            "State": pd.Categorical(["orissa", "Odisha ", "KERALA"]),
            "Commodity": ["ladies finger", "Bhindi", None],
            "Modal_Price": [1.0, 2.0, 3.0],
        }
    )
    out = cleaning.standardise_names(df)
    assert isinstance(out["State"].dtype, pd.CategoricalDtype)
    assert out["State"].astype(str).tolist() == ["Odisha", "Odisha", "Kerala"]
    assert set(out["State"].cat.categories) == {"Odisha", "Kerala"}
    assert out["Commodity"].tolist()[:2] == ["Bhindi(Ladies Finger)"] * 2
    assert pd.isna(out["Commodity"].iloc[2])
    assert out["Modal_Price"].tolist() == [1.0, 2.0, 3.0]
    assert df["State"].astype(str).tolist() == ["orissa", "Odisha ", "KERALA"]


def test_standardise_names_skips_missing_columns() -> None:
    df = pd.DataFrame({"Market": [" pune "]})
    out = cleaning.standardise_names(df, columns=["Market", "District"], aliases={})
    assert out.columns.tolist() == ["Market"]
    assert out["Market"].iloc[0] == "Pune"
