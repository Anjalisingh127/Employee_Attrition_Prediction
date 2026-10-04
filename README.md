# Employee Attrition Prediction & HR Analytics Platform

An end-to-end machine learning project for analyzing employee attrition patterns and estimating individual attrition risk using the IBM HR Analytics Employee Attrition dataset.

## Project status

The repository is being rebuilt from an earlier academic prototype into a reproducible, leakage-safe portfolio project.

**Current milestone: Stage 1.1 — data foundation and validation.**

## Contributors

- Anjali Singh — [@Anjalisingh127](https://github.com/Anjalisingh127)
- Gauri Jakhmola — [@gaurijakhmola](https://github.com/gaurijakhmola)

## Current engineering stack

- Python 3.13+
- pandas
- Pydantic + pydantic-settings
- pytest
- Ruff
- `pyproject.toml` packaging with a `src/` layout
- environment-based configuration through local `.env` files

The committed `.env.example` documents configuration keys. The real `.env` is intentionally ignored and must never contain committed secrets.

## Stage 1.1: dataset contract

The validation layer currently verifies:

- exactly 1,470 rows and 35 columns;
- target values are exactly `Yes` and `No`;
- zero missing values and zero duplicate rows;
- `EmployeeNumber` is unique for all 1,470 records;
- `EmployeeCount`, `Over18`, and `StandardHours` are invariant;
- the source contains 237 attrition cases and 1,233 non-attrition cases.

Those checks turn assumptions about the dataset into executable tests instead of undocumented notebook observations.

## Local setup

Create or reuse a virtual environment, then install the project in editable mode with development tools:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Validate the dataset:

```powershell
attrition-validate
```

Run automated quality checks:

```powershell
python -m pytest
python -m ruff check .
```

## Dataset

The project uses the public IBM HR Analytics Employee Attrition dataset. It contains 1,470 employee records and 35 columns and is a benchmark/synthetic dataset rather than production HR data.

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

See `docs/data_dictionary.md` for the current feature-treatment and leakage policy.

## Target architecture

The finished project will include reproducible preprocessing, business-motivated feature engineering, leakage-safe imbalance handling, Logistic Regression/Random Forest/XGBoost/LightGBM comparisons, stratified cross-validation, hyperparameter tuning, threshold analysis, SHAP explainability, and an interactive Streamlit application.

## Repository roadmap

- **Stage 1 — Data foundation:** dataset audit, schema validation, data dictionary, train/test strategy
- **Stage 2 — EDA:** attrition distribution and workforce risk patterns
- **Stage 3 — Feature engineering:** validated, non-leaking business features
- **Stage 4 — Modeling:** reproducible preprocessing + model baselines
- **Stage 5 — Imbalance experiments:** class weights, oversampling, SMOTE
- **Stage 6 — Tuning & evaluation:** stratified 5-fold CV and final test metrics
- **Stage 7 — Explainability:** feature importance, permutation importance, SHAP
- **Stage 8 — Streamlit application:** workforce analytics and employee risk scoring
- **Stage 9 — Engineering polish:** CI, Docker, documentation, and deployment

## Evaluation principle

Accuracy alone is not sufficient for an imbalanced attrition problem. Final model selection will consider recall, precision, F1, ROC-AUC, PR-AUC, calibration, threshold behavior, and interpretability. No legacy performance claim will be reused unless reproduced through the rebuilt leakage-safe pipeline.

## Repository history

The earlier repository contained downloaded notebooks, generated artifacts, experimental model files, and unrelated email-scanning code from an unfinished prototype. Those files were removed from the current branch while historical commits were preserved.
