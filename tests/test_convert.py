"""Tests for the chunked CSV to Parquet conversion in agromaster.convert."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from agromaster import config, convert

RAW_HEADER = (
    "State,District,Market,Commodity,Variety,Grade,"
    "Arrival_Date,Min Price,Max Price,Modal Price"
)
RAW_ROWS = [
    "Kerala,Ernakulam,Aluva,Banana, Nendra ,FAQ,01/03/2024,3000,3500,3200",
    "Kerala,Ernakulam,Aluva,Banana,Nendra,FAQ,02/03/2024,3100,3600,3300",
    "Punjab,Ludhiana,Khanna,Wheat,Other,FAQ,02/03/2024,2200,2400,2300",
    "Punjab,Ludhiana,Khanna,Wheat,Other,FAQ,03/03/2024,2250,2450,",
    "Punjab,Ludhiana,Khanna,Wheat,Other,FAQ,04/03/2024,2260,2460,2360",
]


def write_csv(path: Path, rows: list[str] = RAW_ROWS, header: str = RAW_HEADER) -> Path:
    """Write a tiny raw-style CSV and return its path."""
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Modal Price", "Modal_Price"),
        ("modal_price", "Modal_Price"),
        ("Modal_x0020_Price", "Modal_Price"),
        (" Arrival_Date ", config.DATE_COLUMN),
        ("Date", config.DATE_COLUMN),
        ("STATE", "State"),
        ("Some Extra Col", "some_extra_col"),
    ],
)
def test_normalise_column_name(raw: str, expected: str) -> None:
    assert convert.normalise_column_name(raw) == expected


def test_infer_date_format_picks_matching_format() -> None:
    values = pd.Series(["2024-03-01", "2024-03-02", None, ""])
    assert convert.infer_date_format(values) == "%Y-%m-%d"
    assert convert.infer_date_format(pd.Series([None, ""])) is None


def test_parse_dates_day_first_and_bad_values() -> None:
    parsed = convert.parse_dates(pd.Series(["01/03/2024", "13/03/2024", "bad"]))
    assert parsed.iloc[0] == pd.Timestamp("2024-03-01")
    assert parsed.iloc[1] == pd.Timestamp("2024-03-13")
    assert pd.isna(parsed.iloc[2])


def test_cast_chunk_dtypes(tmp_path: Path) -> None:
    raw = pd.read_csv(write_csv(tmp_path / "raw.csv"), dtype=str)
    out = convert.cast_chunk(raw)
    for col in config.CATEGORICAL_COLUMNS:
        assert isinstance(out[col].dtype, pd.CategoricalDtype)
    for col in config.PRICE_COLUMNS:
        assert out[col].dtype == "float32"
    assert pd.api.types.is_datetime64_any_dtype(out[config.DATE_COLUMN])
    # Category values are trimmed, so " Nendra " and "Nendra" are one variety.
    assert set(out["Variety"].cat.categories) == {"Nendra", "Other"}
    assert pd.isna(out["Modal_Price"].iloc[3])


def test_cast_chunk_missing_column_raises() -> None:
    raw = pd.DataFrame({"State": ["Kerala"], "Modal Price": ["10"]})
    with pytest.raises(KeyError, match="missing expected columns"):
        convert.cast_chunk(raw)


def test_read_csv_chunks_splits_rows(tmp_path: Path) -> None:
    chunks = list(convert.read_csv_chunks(write_csv(tmp_path / "raw.csv"), chunk_rows=2))
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_convert_writes_parquet_in_chunks(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "raw.csv")
    dest = tmp_path / "interim" / "prices.parquet"
    rows = convert.convert_csv_to_parquet(src=src, dest=dest, chunk_rows=2)

    assert rows == len(RAW_ROWS)
    assert dest.exists()
    assert not dest.with_suffix(".parquet.tmp").exists()
    assert pq.ParquetFile(dest).num_row_groups == 3

    df = pd.read_parquet(dest)
    assert len(df) == len(RAW_ROWS)
    assert list(df.columns) == [
        *config.SERIES_KEYS,
        config.DATE_COLUMN,
        *config.PRICE_COLUMNS,
    ]
    for col in config.PRICE_COLUMNS:
        assert df[col].dtype == "float32"
    for col in config.CATEGORICAL_COLUMNS:
        assert isinstance(df[col].dtype, pd.CategoricalDtype)
    assert df[config.DATE_COLUMN].min() == pd.Timestamp("2024-03-01")
    assert df[config.DATE_COLUMN].max() == pd.Timestamp("2024-03-04")
    assert df["Modal_Price"].isna().sum() == 1


def test_convert_header_only_csv_raises(tmp_path: Path) -> None:
    src = write_csv(tmp_path / "empty.csv", rows=[])
    dest = tmp_path / "prices.parquet"
    with pytest.raises(ValueError, match="no data rows"):
        convert.convert_csv_to_parquet(src=src, dest=dest)
    assert not dest.exists()


def test_find_raw_csv_picks_largest(tmp_path: Path) -> None:
    (tmp_path / "small.csv").write_text("a\n1\n")
    (tmp_path / "sub").mkdir()
    big = tmp_path / "sub" / "big.csv"
    big.write_text("a\n" + "1\n" * 100)
    assert convert.find_raw_csv(tmp_path) == big


def test_find_raw_csv_without_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download step"):
        convert.find_raw_csv(tmp_path)
