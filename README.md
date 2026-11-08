# AgroMaster: Agricultural Market Price Decline Risk

EC5203 Machine Learning mini project, Group 41, University of Ruhuna.

The model classifies agricultural market price-decline risk as **Low**, **Medium** or **High** seven days ahead, using AGMARKNET mandi prices. It compares Random Forest and XGBoost against baselines. See [proposal/main.tex](proposal/main.tex) for the full methodology.

## Setup

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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

## Pipeline stages

1. Download and convert the CSV to Parquet in chunks
2. EDA and modelling subset selection
3. Cleaning
4. Features (18) and 7-day risk labels
5. Baselines, Random Forest, XGBoost with chronological CV
6. Evaluation and SHAP interpretation
