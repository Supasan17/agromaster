"""Project paths and constants taken from the proposal."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

KAGGLE_DATASET = "khandelwalmanas/daily-commodity-prices-india"

SEED = 42

# Target definition
HORIZON_DAYS = 7
HISTORY_DAYS = 90
HIGH_RISK_QUANTILE = 0.10
MEDIUM_RISK_QUANTILE = 0.30

# Preprocessing
MAX_FFILL_DAYS = 3

# Series identity (used for grouping, not as model inputs)
SERIES_KEYS = ["State", "District", "Market", "Commodity", "Variety", "Grade"]

# Raw CSV columns after name normalisation
DATE_COLUMN = "Arrival_Date"
PRICE_COLUMNS = ["Min_Price", "Max_Price", "Modal_Price"]
CATEGORICAL_COLUMNS = SERIES_KEYS

# CSV to Parquet conversion
CSV_CHUNK_ROWS = 1_000_000
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"]
INTERIM_PARQUET = DATA_INTERIM / "prices.parquet"

# Chronological split (to be confirmed after EDA)
DEV_END = "2023-12-31"
TEST_START = "2024-01-01"
N_CV_FOLDS = 5
