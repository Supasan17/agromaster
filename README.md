# AgroMaster: Agricultural Market Price Decline Risk

EC5203 Machine Learning mini project, Group 41, University of Ruhuna.

The model classifies agricultural market price-decline risk as **Low**, **Medium** or **High** seven days ahead, using AGMARKNET mandi prices. It compares Random Forest and XGBoost against baselines. See [proposal/main.tex](proposal/main.tex) for the full methodology.

## Setup

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

To download the data, put your Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`. Never commit it.

## Layout

| Path | Contents |
|---|---|
| `data/raw` | Original Kaggle CSV (git-ignored) |
| `data/interim` | Chunk-converted Parquet (git-ignored) |
| `data/processed` | Cleaned subset with features and labels (git-ignored) |
| `notebooks` | EDA and experiments, numbered `01_...`, `02_...` |
| `src/agromaster` | Reusable pipeline code (config, cleaning, features, labels, models) |
| `models` | Saved trained models (git-ignored) |
| `reports/figures` | Plots for the report and presentation |
| `proposal` | LaTeX proposal |

## Running the pipeline

The pipeline runs from the command line, one stage or several at a time:

```powershell
python -m agromaster <stage> [<stage> ...]
python -m agromaster all
```

Each stage reads the output of the stage before it, so run them in this order (`all` runs every stage in order):

| Stage | What it does | Output |
|---|---|---|
| `download` | Downloads the Kaggle dataset (skipped if a CSV is already there). Needs the Kaggle token. | `data/raw/` |
| `convert` | Reads the CSV in chunks, normalises column names and types, parses dates | `data/interim/prices.parquet` |
| `clean` | Drops duplicates, missing, non-positive and inconsistent prices, standardises names, takes the median of same-day records and aligns each series to calendar days (predictors forward-filled at most 3 days, targets never filled) | `data/processed/clean.parquet` |
| `features` | Adds the predictors (price spread, calendar and season, lags, rolling means, volatility, momentum) and the 7-day ahead price ratio | `data/processed/features.parquet` |
| `train` | Splits chronologically (development to 2023-12-31, test from 2024-01-01, rows whose t+7 crosses the boundary dropped), fits the commodity risk thresholds on the development period only, then fits the majority, logistic regression, Random Forest and XGBoost models | `models/risk_thresholds.parquet`, `models/<model>.joblib` |
| `evaluate` | Labels the test period with the development thresholds unchanged and scores every model, including the risk-persistence baseline | `reports/*.csv` |

`evaluate` writes `reports/metrics_summary.csv` (accuracy, macro precision, recall and F1, and High-risk recall for each model) and, for each model, the per-class metrics, the confusion matrix and the per-commodity and per-year reports.

If a stage's input is missing, the command stops with a message naming the stage to run first. Constants such as the horizon, thresholds, split dates and model settings are in `src/agromaster/config.py`.

## Tests and linting

```powershell
python -m pytest -q
python -m ruff check src tests
```

The tests build small synthetic DataFrames, so they need neither the real data nor the Kaggle token. GitHub Actions runs both checks on every push and pull request (`.github/workflows/ci.yml`).
