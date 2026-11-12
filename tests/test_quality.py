"""Tests for the data quality summaries in agromaster.quality."""
import pandas as pd

from agromaster import config, quality


def make_prices() -> pd.DataFrame:
    """Small interim-style price frame with a few nulls and an unused category."""
    return pd.DataFrame(
        {
            "State": pd.Categorical(
                ["Kerala", "Kerala", "Punjab", "Kerala"],
                categories=["Kerala", "Punjab", "Goa"],
            ),
            "Commodity": pd.Categorical(["Banana", "Banana", "Wheat", "Rice"]),
            config.DATE_COLUMN: pd.to_datetime(
                ["2024-03-01", "2024-03-03", None, "2024-03-03"]
            ),
            "Modal_Price": pd.Series([3200.0, None, 2300.0, None], dtype="float32"),
        }
    )


def test_row_count() -> None:
    assert quality.row_count(make_prices())["rows"].iloc[0] == 4


def test_null_counts_per_column() -> None:
    nulls = quality.null_counts(make_prices()).set_index("column")
    assert list(nulls.index) == ["State", "Commodity", config.DATE_COLUMN, "Modal_Price"]
    assert nulls.loc["State", "nulls"] == 0
    assert nulls.loc[config.DATE_COLUMN, "nulls"] == 1
    assert nulls.loc["Modal_Price", "nulls"] == 2
    assert nulls.loc["Modal_Price", "null_share"] == 0.5


def test_date_range_ignores_missing_dates() -> None:
    out = quality.date_range(make_prices()).iloc[0]
    assert out["first_date"] == pd.Timestamp("2024-03-01")
    assert out["last_date"] == pd.Timestamp("2024-03-03")
    assert out["span_days"] == 3
    assert out["distinct_dates"] == 2


def test_records_per_commodity_sorted() -> None:
    out = quality.records_per_commodity(make_prices())
    assert list(out.columns) == ["Commodity", "records"]
    assert out.iloc[0].tolist() == ["Banana", 2]
    assert out["records"].sum() == 4


def test_records_per_state_drops_unused_categories() -> None:
    out = quality.records_per_state(make_prices())
    assert dict(zip(out["State"].astype(str), out["records"])) == {"Kerala": 3, "Punjab": 1}


def test_summary_on_empty_frame() -> None:
    summary = quality.quality_summary(make_prices().iloc[:0])
    assert set(summary) == {"rows", "nulls", "date_range", "commodity", "state"}
    assert summary["rows"]["rows"].iloc[0] == 0
    assert (summary["nulls"]["null_share"] == 0).all()
    assert summary["date_range"]["span_days"].iloc[0] == 0
    assert pd.isna(summary["date_range"]["first_date"].iloc[0])
    assert summary["commodity"].empty
