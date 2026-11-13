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
# Calendar alignment keeps the observed prices (used for targets) unfilled and
# adds forward-filled copies, named with this suffix, for use as predictors.
FILLED_SUFFIX = "_Filled"
OBSERVED_COLUMN = "Is_Observed"

# Series identity (used for grouping, not as model inputs)
SERIES_KEYS = ["State", "District", "Market", "Commodity", "Variety", "Grade"]

# Raw CSV columns after name normalisation
DATE_COLUMN = "Arrival_Date"
PRICE_COLUMNS = ["Min_Price", "Max_Price", "Modal_Price"]
CATEGORICAL_COLUMNS = SERIES_KEYS

# Name standardisation: values in these columns are trimmed, have repeated
# spaces collapsed and are case-folded; known spelling variants (keyed by the
# case-folded form) are then mapped to one canonical name.
NAME_COLUMNS = SERIES_KEYS
NAME_ALIASES: dict[str, dict[str, str]] = {
    "State": {
        "chattisgarh": "Chhattisgarh",
        "orissa": "Odisha",
        "pondicherry": "Puducherry",
        "uttrakhand": "Uttarakhand",
        "jammu & kashmir": "Jammu and Kashmir",
        "nct of delhi": "Delhi",
    },
    "Commodity": {
        "bhindi": "Bhindi(Ladies Finger)",
        "ladies finger": "Bhindi(Ladies Finger)",
    },
}

# CSV to Parquet conversion
CSV_CHUNK_ROWS = 1_000_000
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"]
INTERIM_PARQUET = DATA_INTERIM / "prices.parquet"

# Chronological split (to be confirmed after EDA)
DEV_END = "2023-12-31"
TEST_START = "2024-01-01"
N_CV_FOLDS = 5
