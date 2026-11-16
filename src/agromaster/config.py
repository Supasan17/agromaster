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
# The label ratio is the observed modal price HORIZON_DAYS calendar days after
# the prediction date t divided by the mean observed modal price over the
# HISTORY_DAYS calendar days before t (t itself excluded). The history mean
# needs at least this many observed prices.
HISTORY_MIN_PERIODS = 1
HISTORY_MEAN_COLUMN = "History_Mean_90"
FUTURE_PRICE_COLUMN = "Modal_Future_7"
RATIO_COLUMN = "Future_Ratio"
# Risk labels are assigned per commodity from thresholds fitted on a training
# frame: ratio <= HIGH_RISK_QUANTILE threshold is High, ratio <=
# MEDIUM_RISK_QUANTILE threshold is Medium, anything above is Low.
COMMODITY_COLUMN = "Commodity"
RISK_LABELS = ["Low", "Medium", "High"]
LABEL_COLUMN = "Risk_Label"
HIGH_THRESHOLD_COLUMN = "High_Threshold"
MEDIUM_THRESHOLD_COLUMN = "Medium_Threshold"

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

# Feature names (proposal feature table)
SPREAD_COLUMN = "Price_Spread"
SPREAD_RATIO_COLUMN = "Spread_Ratio"
MONTH_COLUMN = "Month"
YEAR_COLUMN = "Year"
DAY_OF_YEAR_COLUMN = "Day_of_Year"
SEASON_COLUMN = "Season"

# Lag features (features 8 to 10): the modal price this many calendar days
# before the prediction date, within the same series.
LAG_DAYS = [1, 7, 30]
LAG_COLUMN_TEMPLATE = "Modal_Lag_{days}"

# Rolling-window features (features 11 to 14). Windows are calendar days
# ending on (and including) the prediction date, within the same series, so
# they use no information after that date.
MA_WINDOWS = [7, 30]
MA_COLUMN_TEMPLATE = "MA_{days}"
VOLATILITY_WINDOW = 30
VOLATILITY_COLUMN = "Volatility_30"
# Price momentum is the percentage change (in percent) of the modal price
# over this many calendar days.
MOMENTUM_DAYS = 7
MOMENTUM_COLUMN = "Price_Momentum"
# Minimum number of observed prices in a window for the statistic to be
# reported (a standard deviation needs at least two).
MA_MIN_PERIODS = 1
VOLATILITY_MIN_PERIODS = 2

# Month-to-season mapping for India, following the four seasons used by the
# India Meteorological Department (IMD): Winter (January-February), Summer or
# pre-monsoon (March-May), Monsoon or south-west monsoon (June-September) and
# Post-monsoon (October-December).
SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
SEASON_BY_MONTH: dict[int, str] = {
    1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-monsoon", 11: "Post-monsoon", 12: "Post-monsoon",
}

# CSV to Parquet conversion
CSV_CHUNK_ROWS = 1_000_000
DATE_FORMATS = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"]
INTERIM_PARQUET = DATA_INTERIM / "prices.parquet"

# Chronological split (to be confirmed after EDA)
DEV_END = "2023-12-31"
TEST_START = "2024-01-01"
N_CV_FOLDS = 5
