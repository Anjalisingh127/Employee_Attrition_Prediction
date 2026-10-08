# Employee Attrition Prediction & HR Analytics Platform

An end-to-end machine learning project for analyzing employee attrition patterns and estimating individual attrition risk using the IBM HR Analytics Employee Attrition dataset.

## Project status

The repository is being rebuilt from an earlier academic prototype into a reproducible, leakage-safe portfolio project.

**Current milestone: Stage 7.4 — SHAP local explainability.**

## Contributors

- Anjali Singh — [@Anjalisingh127](https://github.com/Anjalisingh127)
- Gauri Jakhmola — [@gaurijakhmola](https://github.com/gaurijakhmola)

## Current engineering stack

- Python 3.13+
- pandas
- scikit-learn
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

## Stage 1.2: modeling foundation

Before model development, the project now creates a deterministic 80/20 stratified holdout and removes the target, employee identifier, and three invariant columns from model inputs. This leaves 30 candidate predictors.

The final test partition is reserved for final evaluation. Encoding, scaling, resampling, feature selection, hyperparameter tuning, and threshold selection must be learned from training data only.

See `docs/modeling_protocol.md` for the enforced evaluation rules.

## Stage 2.1: reproducible EDA

Exploratory analysis is implemented as reusable Python functions rather than notebook-only calculations. The first report covers overall attrition, business segment rates, and median comparisons for selected numeric workforce features.

Small groups can be filtered with a minimum-size threshold, and every result is explicitly treated as a descriptive association rather than causal evidence.

Run the machine-readable report with:

```powershell
attrition-eda
```

See `docs/eda_protocol.md` for the analysis and interpretation rules.

## Stage 2.2: visual analysis and findings

The project now generates a focused set of presentation-ready EDA charts directly from the validated source data. Generated images are intentionally excluded from Git so the repository stores the code and evidence needed to reproduce them rather than stale binary outputs.

```powershell
attrition-eda-charts
```

The documented findings in `docs/eda_findings.md` highlight verified associations around overtime, job role, business travel, satisfaction, work-life balance, and selected numeric workforce measures. They are not presented as causal effects.

## Stage 3: feature engineering and preprocessing

Five deterministic business features capture compensation level, career/company tenure, role tenure, promotion timing, and early-career status without using the target. They are documented in `docs/feature_engineering.md`.

The preprocessing layer then combines those features with training-fitted numeric scaling and one-hot categorical encoding. It is intentionally returned unfitted and will be placed inside every Stage 4 model pipeline so cross-validation learns preprocessing independently within each fold.

See `docs/preprocessing.md` for the schema and leakage controls.

## Stage 4.1: baseline modeling

The first predictive benchmark compares a class-prior dummy baseline, Logistic Regression, and Random Forest using training-only Stratified 5-Fold Cross-Validation. Every estimator contains the complete Stage 3 preprocessing workflow, so learned preprocessing is refitted independently inside each fold.

Run the reproducible baseline report with:

```powershell
attrition-baselines
```

The report includes accuracy, precision, recall, F1, ROC-AUC, and PR-AUC with fold-level values and aggregate variation. The 294-row final test partition remains untouched.

See `docs/baseline_modeling.md` for the evaluation protocol.

## Stage 4.2: comparison and diagnostics

Stage 4.1 results now feed a reproducible comparison layer that ranks baselines by PR-AUC, then recall and ROC-AUC, without using an arbitrary weighted score. It also reports fold-level stability through standard deviation and observed min/max ranges.

Run:

```powershell
attrition-compare
```

The command regenerates a CSV comparison table and JSON diagnostics under `reports/generated/`. These artifacts are ignored by Git and can always be reproduced from the validated training workflow. The 294-row final holdout is still not evaluated.

See `docs/baseline_diagnostics.md` for the ranking and interpretation policy.

## Stage 5: imbalance handling

The two predictive baselines now enter controlled imbalance experiments using class weighting, RandomOverSampler, and SMOTE. Resampling is implemented with an imbalanced-learn pipeline after preprocessing and occurs only inside each cross-validation training fold.

Run:

```powershell
attrition-imbalance
```

The command compares six Logistic Regression/Random Forest strategy combinations using the same Stratified 5-Fold CV and metrics as Stage 4. The final 294-row holdout remains untouched.

See `docs/imbalance_experiments.md` for the experimental protocol.

## Stage 6.1: hyperparameter tuning

The three strongest Stage 5 candidates now enter a two-pass tuning workflow: bounded `RandomizedSearchCV` followed by a targeted `GridSearchCV` around each randomized-search winner. PR-AUC is the refit metric, and every search uses shuffled Stratified 5-Fold CV on the 1,176-row training partition only.

Run:

```powershell
attrition-tune
```

The final 294-row holdout remains untouched. See `docs/hyperparameter_tuning.md` for search spaces, leakage controls, and the refinement strategy.

## Stage 6.2: threshold analysis and calibration

The tuned Logistic Regression model is now evaluated with out-of-fold probabilities to select a decision threshold without touching the final test set. The project also compares uncalibrated, sigmoid-calibrated, and isotonic-calibrated probabilities using Brier score and log loss.

Run:

```powershell
attrition-calibrate
```

The threshold policy maximizes F1 without inventing business cost weights, while calibration is selected by lowest out-of-fold Brier score. The final 294-row holdout remains untouched.

See `docs/threshold_calibration.md` for the full methodology.

## Stage 6.3: final holdout evaluation

The model policy is now frozen: tuned Logistic Regression (`C=0.3`, L2), isotonic calibration, and decision threshold `0.38`. Stage 6.3 fits that frozen policy on the full training partition and evaluates the reserved 294-row holdout exactly once.

Run:

```powershell
attrition-final-eval
```

The final report includes threshold-dependent classification metrics, ROC-AUC, PR-AUC, Brier score, log loss, and confusion-matrix counts. After observing these results, the project will not modify the model based on the holdout.

See `docs/final_holdout_evaluation.md` for the final-evaluation protocol.

## Stage 7.1: Logistic Regression coefficient analysis

The frozen tuned Logistic Regression now has a reproducible global coefficient analysis. The explanation maps the transformed preprocessing feature names directly to the fitted coefficient vector and ranks features by positive/negative contribution to attrition log-odds.

Run:

```powershell
attrition-coefficients
```

The report includes coefficients, odds ratios, absolute magnitudes, and direction. These are model explanations rather than causal HR effects. The consumed final holdout is excluded from this explainability fit.

See `docs/coefficient_analysis.md` for interpretation guidance and limitations.

## Stage 7.2: permutation importance

The frozen calibrated prediction policy now has a model-agnostic global importance analysis. Original input features are shuffled on held-out folds of the training partition, and importance is measured by the resulting drop in Average Precision and ROC-AUC.

Run:

```powershell
attrition-permutation
```

This complements Stage 7.1 coefficient analysis by measuring predictive reliance on original business features through the complete pipeline. The consumed final holdout is not reused.

See `docs/permutation_importance.md` for methodology and interpretation limits.

## Stage 7.3: SHAP global explainability

The frozen tuned Logistic Regression now has a global SHAP explanation layer using `shap.LinearExplainer`. SHAP values are computed on the transformed training feature matrix and aggregated by mean absolute contribution while preserving signed row-level effects.

Run:

```powershell
attrition-shap-global
```

The stage generates a ranked SHAP table plus global bar and beeswarm plots. It explains the underlying Logistic Regression decision function; isotonic calibration remains a separate probability post-processing layer. The consumed final holdout is not reused.

See `docs/shap_global.md` for methodology and limitations.

## Stage 7.4: SHAP local explainability

The project now supports individual training-partition explanations. For a selected row, it reports the frozen calibrated probability and threshold decision while separately decomposing the underlying Logistic Regression score into risk-increasing and risk-reducing SHAP contributions.

Run the default example:

```powershell
attrition-shap-local
```

Or choose a deterministic training-row position:

```powershell
attrition-shap-local --row-position 25 --top-n 8
```

The command produces a ranked contribution table, JSON summary, and SHAP waterfall plot. These explanations are model-specific, non-causal, and do not reuse the consumed final holdout.

See `docs/shap_local.md` for methodology and interpretation guidance.

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
