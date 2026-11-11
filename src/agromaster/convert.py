"""Convert the raw AGMARKNET CSV to compressed Parquet in data/interim/, chunk by chunk.

The source file has more than 75 million rows, so it is never loaded whole.
Each chunk gets normalised column names, categorical location and product
columns, float32 prices and parsed arrival dates, and is appended to one
Parquet file as its own row group.
"""
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from agromaster import config

# Lower-cased, underscore-joined raw names mapped to the canonical names in config.
COLUMN_ALIASES: dict[str, str] = {
    "state": "State",
    "district": "District",
    "market": "Market",
    "commodity": "Commodity",
    "variety": "Variety",
    "grade": "Grade",
    "arrival_date": config.DATE_COLUMN,
    "date": config.DATE_COLUMN,
    "min_price": "Min_Price",
    "max_price": "Max_Price",
    "modal_price": "Modal_Price",
    "min_x0020_price": "Min_Price",
    "max_x0020_price": "Max_Price",
    "modal_x0020_price": "Modal_Price",
}


def normalise_column_name(name: str) -> str:
    """Map one raw header to its canonical name.

    Headers are trimmed, lower-cased and joined with underscores (so
    ``"Modal Price"``, ``"modal_price"`` and ``"Modal_x0020_Price"`` all match).
    Unknown headers keep that cleaned form.
    """
    key = re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")
    return COLUMN_ALIASES.get(key, key)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical column names."""
    return df.rename(columns={c: normalise_column_name(c) for c in df.columns})


def infer_date_format(values: pd.Series, formats: list[str] | None = None) -> str | None:
    """Return the format in ``formats`` that parses the most non-null values.

    Ties go to the earlier format, so day-first formats win over month-first
    ones. A few malformed values therefore do not push the whole chunk onto a
    wrong format. Returns ``None`` if no format parses any value.
    """
    sample = values.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(1000)
    if sample.empty:
        return None
    best_fmt, best_count = None, 0
    for fmt in formats or config.DATE_FORMATS:
        count = int(pd.to_datetime(sample, format=fmt, errors="coerce").notna().sum())
        if count > best_count:
            best_fmt, best_count = fmt, count
    return best_fmt


def parse_dates(values: pd.Series, formats: list[str] | None = None) -> pd.Series:
    """Parse arrival dates; values that do not match the inferred format become NaT."""
    fmt = infer_date_format(values, formats)
    if fmt is None:
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values.astype(str).str.strip(), format=fmt, errors="coerce")


def cast_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise names and cast categoricals, float32 prices and dates."""
    df = normalise_columns(df)
    missing = [
        c
        for c in [*config.CATEGORICAL_COLUMNS, config.DATE_COLUMN, *config.PRICE_COLUMNS]
        if c not in df.columns
    ]
    if missing:
        raise KeyError(f"Raw CSV is missing expected columns: {missing}")
    out = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("string").str.strip().astype("category")
    for col in config.PRICE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out[config.DATE_COLUMN] = parse_dates(out[config.DATE_COLUMN])
    return out


def arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """Fixed Parquet schema so every chunk's row group matches.

    Categorical columns are stored as dictionary-encoded strings, so each
    chunk can carry its own category set. Any other column (read from the
    CSV as text) is kept as a plain string.
    """
    fields = []
    for col in df.columns:
        if col in config.CATEGORICAL_COLUMNS:
            fields.append(pa.field(col, pa.dictionary(pa.int32(), pa.string())))
        elif col in config.PRICE_COLUMNS:
            fields.append(pa.field(col, pa.float32()))
        elif col == config.DATE_COLUMN:
            fields.append(pa.field(col, pa.timestamp("ns")))
        else:
            fields.append(pa.field(col, pa.string()))
    return pa.schema(fields)


def read_csv_chunks(path: Path, chunk_rows: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield the CSV in chunks of ``chunk_rows`` rows, all columns read as strings."""
    yield from pd.read_csv(
        path,
        chunksize=chunk_rows or config.CSV_CHUNK_ROWS,
        dtype=str,
        keep_default_na=True,
    )


def find_raw_csv(raw_dir: Path | None = None) -> Path:
    """Return the largest CSV under ``raw_dir`` (the main price file)."""
    raw_dir = raw_dir or config.DATA_RAW
    csvs = sorted(raw_dir.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Run the download step first."
        )
    return csvs[0]


def convert_csv_to_parquet(
    src: Path | None = None,
    dest: Path | None = None,
    chunk_rows: int | None = None,
) -> int:
    """Convert ``src`` CSV to Parquet at ``dest`` chunk by chunk; return rows written."""
    src = src or find_raw_csv()
    dest = dest or config.INTERIM_PARQUET
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows = 0
    try:
        for chunk in read_csv_chunks(src, chunk_rows):
            chunk = cast_chunk(chunk)
            if writer is None:
                schema = arrow_schema(chunk)
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{src} has no data rows")
    tmp.replace(dest)
    print(f"Wrote {rows:,} rows to {dest}")
    return rows


if __name__ == "__main__":
    convert_csv_to_parquet()
