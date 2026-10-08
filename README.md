# Employee Attrition Prediction & HR Analytics Platform

A production-style machine learning project for analyzing employee attrition patterns, estimating attrition risk, and explaining model behavior using the IBM HR Analytics Employee Attrition dataset.

The project is designed around reproducibility, leakage-safe evaluation, calibrated risk estimation, and interpretable predictions rather than relying on a single accuracy score.

---

## Overview

The system implements an end-to-end employee attrition modeling workflow covering:

- automated dataset validation;
- deterministic stratified train/holdout splitting;
- business-oriented feature engineering;
- reusable preprocessing pipelines;
- Logistic Regression and Random Forest baselines;
- class weighting, RandomOverSampler, and SMOTE experiments;
- stratified cross-validation;
- two-stage hyperparameter tuning;
- probability calibration;
- decision-threshold optimization;
- one-time final holdout evaluation;
- coefficient analysis;
- permutation importance;
- global and local SHAP explainability;
- cross-method explainability reporting;
- automated tests and static code-quality checks.

The current model and explainability workflow is complete. The next development phase is the interactive Streamlit application and deployment engineering.

---

## Key Results

### Final model

| Component | Configuration |
| --- | --- |
| Model | Logistic Regression |
| Regularization | L2 |
| C | 0.3 |
| Solver | `liblinear` |
| Probability calibration | Isotonic |
| Calibration CV | 3 folds |
| Decision threshold | 0.38 |

The model policy was selected entirely from the training partition before the final holdout was evaluated.

### Final holdout performance

The frozen policy was evaluated once on a reserved 294-row holdout set.

| Metric | Result |
| --- | ---: |
| Accuracy | **87.07%** |
| Precision | **62.16%** |
| Recall | **48.94%** |
| F1 Score | **54.76%** |
| ROC-AUC | **81.08%** |
| Average Precision | **58.39%** |
| Brier Score | **0.0967** |
| Log Loss | **0.4393** |

### Confusion matrix

|  | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| Actual No | **233** | **14** |
| Actual Yes | **24** | **23** |

> Some internal reports use the historical key name `pr_auc` while calling
> `average_precision_score`. In this README, that metric is referred to correctly as
> **Average Precision (AP)**.

No model, preprocessing, calibration, or threshold changes are made using the observed holdout results.

---

## System Architecture

```text
┌─────────────────────────────────────┐
│ IBM HR Analytics Dataset            │
│ 1,470 employee records              │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Data Validation                     │
│                                     │
│ • schema checks                     │
│ • missing / duplicate checks        │
│ • target validation                 │
│ • invariant-column validation       │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Stratified Train / Holdout Split    │
│                                     │
│ Training: 1,176 rows                │
│ Final holdout: 294 rows             │
└───────────────┬─────────────────────┘
                │
                │ training partition only
                v
┌─────────────────────────────────────┐
│ Feature Engineering                 │
│                                     │
│ • IncomePerJobLevel                 │
│ • CompanyTenureRatio                │
│ • RoleTenureRatio                   │
│ • PromotionWaitRatio                │
│ • EarlyCareer                       │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Preprocessing Pipeline              │
│                                     │
│ • numeric scaling                   │
│ • categorical one-hot encoding      │
│ • fold-local fitted transformations │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Model Development                   │
│                                     │
│ • Dummy baseline                    │
│ • Logistic Regression               │
│ • Random Forest                     │
│ • class weighting                   │
│ • RandomOverSampler                 │
│ • SMOTE                             │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Training Evaluation & Tuning        │
│                                     │
│ • Stratified 5-Fold CV              │
│ • RandomizedSearchCV                │
│ • GridSearchCV                      │
│ • Average Precision / ROC-AUC       │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Calibration & Threshold Selection   │
│                                     │
│ • sigmoid vs isotonic               │
│ • Brier score / log loss            │
│ • threshold optimization            │
└──────────────────┬──────────────────┘
                   │
                   v
┌─────────────────────────────────────┐
│ Frozen Prediction Policy            │
│                                     │
│ Logistic Regression                 │
│ C = 0.3, L2, liblinear              │
│ Isotonic calibration                │
│ Threshold = 0.38                    │
└───────────────┬─────────────────────┘
                │
                ├──────────────────────────────┐
                │                              │
                v                              v
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ Final Holdout Evaluation     │   │ Explainability              │
│                              │   │                              │
│ 294 unseen rows              │   │ • coefficients              │
│ one-time evaluation          │   │ • permutation importance    │
│                              │   │ • global SHAP               │
└──────────────────────────────┘   │ • local SHAP                │
                                   │ • consensus reporting        │
                                   └──────────────┬───────────────┘
                                                  │
                                                  v
                                   ┌──────────────────────────────┐
                                   │ Application Layer            │
                                   │                              │
                                   │ Streamlit dashboard          │
                                   │ risk prediction              │
                                   │ explainable results          │
                                   │                              │
                                   │ planned                      │
                                   └──────────────────────────────┘
```

### Architecture principles

- **Leakage prevention:** learned preprocessing, resampling, calibration, tuning, and threshold selection are performed using training data only.
- **Reproducibility:** major workflows are implemented as Python modules and CLI commands rather than notebook-only steps.
- **Separation of concerns:** validation, feature engineering, modeling, evaluation, calibration, and explainability are independent modules.
- **Frozen evaluation policy:** the final holdout is not reused for post-evaluation model selection.
- **Interpretability:** global and local explanations are provided separately from probability calibration.

---

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Validated properties:

- **1,470 rows**
- **35 source columns**
- target: `Attrition`
- 237 attrition cases
- 1,233 non-attrition cases
- approximately **16.12% attrition rate**
- no missing values
- no duplicate rows
- unique `EmployeeNumber`
- invariant `EmployeeCount`, `Over18`, and `StandardHours`

For modeling, the target column, employee identifier, and invariant columns are excluded before learning begins, leaving **30 original model inputs**.

See [`docs/data_dictionary.md`](docs/data_dictionary.md) for feature treatment details.

---

## Evaluation Strategy

The dataset is split before any learned transformation:

```text
Validated dataset
        |
        v
Stratified 80 / 20 split
        |
        +-------------------------+
        |                         |
        v                         v
Training partition          Final holdout
1,176 rows                  294 rows
        |                         |
        | CV / tuning             |
        | calibration             |
        | threshold selection     |
        | explainability          |
        v                         |
Frozen model policy               |
        |                         |
        +------------------------>|
                                  v
                         One-time final evaluation
```

Training-only operations include:

- scaling;
- categorical encoding;
- engineered features;
- resampling;
- cross-validation;
- hyperparameter search;
- calibration selection;
- threshold selection;
- explainability analysis.

See [`docs/modeling_protocol.md`](docs/modeling_protocol.md).

---

## Feature Engineering & Preprocessing

Five deterministic business features are generated without using the target:

| Feature | Purpose |
| --- | --- |
| `IncomePerJobLevel` | compensation relative to job level |
| `CompanyTenureRatio` | company tenure relative to total career tenure |
| `RoleTenureRatio` | time in current role relative to company tenure |
| `PromotionWaitRatio` | time since promotion relative to company tenure |
| `EarlyCareer` | identifies employees with fewer than five working years |

The preprocessing pipeline applies:

- numeric standardization;
- categorical one-hot encoding with unknown-category handling;
- business feature generation;
- fold-local fitting during cross-validation.

See:

- [`docs/feature_engineering.md`](docs/feature_engineering.md)
- [`docs/preprocessing.md`](docs/preprocessing.md)

---

## Model Development

### Baseline models

The initial benchmark compared:

- class-prior Dummy Classifier;
- Logistic Regression;
- Random Forest.

All models were evaluated using shuffled Stratified 5-Fold Cross-Validation with preprocessing fitted independently inside each fold.

The untreated Logistic Regression produced the strongest overall baseline balance and became the primary candidate for later tuning.

### Class imbalance experiments

Because attrition represents only about 16% of the dataset, the project evaluated:

- class weighting;
- RandomOverSampler;
- SMOTE.

These were tested with Logistic Regression and Random Forest.

Resampling is performed only inside training folds through an `imbalanced-learn` pipeline.

The experiments showed that class weighting increased recall substantially but reduced precision, F1, and Average Precision relative to the untreated Logistic Regression.

See [`docs/imbalance_experiments.md`](docs/imbalance_experiments.md).

### Hyperparameter tuning

Three model strategies were carried forward:

1. Logistic Regression;
2. class-weighted Logistic Regression;
3. Random Forest + SMOTE.

The tuning workflow uses:

- bounded `RandomizedSearchCV`;
- targeted `GridSearchCV`;
- shuffled Stratified 5-Fold CV;
- training data only.

The selected configuration was:

```text
Logistic Regression
C = 0.3
penalty = l2
solver = liblinear
```

See [`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md).

---

## Probability Calibration & Decision Threshold

The tuned Logistic Regression was evaluated with:

- uncalibrated probabilities;
- sigmoid calibration;
- isotonic calibration.

Calibration was selected using out-of-fold training predictions.

Isotonic calibration produced the lowest Brier score under the predefined selection rule and was frozen for the final policy.

The classification threshold was selected independently using training-only out-of-fold predictions.

```text
Final threshold = 0.38
```

The threshold was selected by maximizing F1 without introducing unsupported business-cost assumptions.

See [`docs/threshold_calibration.md`](docs/threshold_calibration.md).

---

## Model Explainability

The project uses multiple complementary explainability techniques rather than relying on one feature-importance chart.

### Logistic Regression coefficients

Coefficient analysis provides signed model direction and odds-ratio interpretation.

Stronger higher modeled attrition log-odds terms include:

- `JobRole_Laboratory Technician`
- `OverTime_Yes`
- `BusinessTravel_Travel_Frequently`
- `NumCompaniesWorked`
- `YearsSinceLastPromotion`

Stronger lower modeled attrition log-odds terms include:

- `OverTime_No`
- `BusinessTravel_Non-Travel`
- `Department_Research & Development`
- `JobRole_Research Director`
- `EducationField_Other`

See [`docs/coefficient_analysis.md`](docs/coefficient_analysis.md).

### Cross-validated permutation importance

Permutation importance measures performance degradation when an original input feature is shuffled on held-out training folds.

Cross-validated performance during this analysis:

- Average Precision: **0.6695 ± 0.0527**
- ROC-AUC: **0.8412 ± 0.0276**

Strong original-feature dependencies include:

- `YearsInCurrentRole`
- `OverTime`
- `TotalWorkingYears`
- `YearsAtCompany`
- `NumCompaniesWorked`
- `MonthlyIncome`
- `YearsSinceLastPromotion`
- `EnvironmentSatisfaction`
- `BusinessTravel`
- `JobRole`

See [`docs/permutation_importance.md`](docs/permutation_importance.md).

### Global SHAP

Global SHAP explanations are generated using `shap.LinearExplainer` on the frozen Logistic Regression decision function.

Strong global SHAP contributors include:

- `OverTime_No`
- `NumCompaniesWorked`
- `EnvironmentSatisfaction`
- `RoleTenureRatio`
- `JobSatisfaction`
- `YearsSinceLastPromotion`
- `DistanceFromHome`
- `JobInvolvement`
- `Age`
- `YearsWithCurrManager`
- `TotalWorkingYears`

Generated artifacts include:

- SHAP global importance table;
- global bar plot;
- SHAP beeswarm plot.

See [`docs/shap_global.md`](docs/shap_global.md).

### Local SHAP

Individual training-partition predictions can be decomposed into:

- calibrated attrition probability;
- threshold decision;
- risk-increasing contributions;
- risk-reducing contributions;
- SHAP waterfall visualization;
- numerical additivity check.

SHAP explains the underlying Logistic Regression score. Isotonic calibration is treated as a separate probability-mapping layer.

See [`docs/shap_local.md`](docs/shap_local.md).

### Cross-method consensus

Global coefficient, permutation, and SHAP rankings are compared at the original feature-family level.

No arbitrary weighted composite score is used.

Features supported in the top tier by **all three global explanation methods** include:

- `OverTime`
- `JobRole`
- `BusinessTravel`
- `NumCompaniesWorked`
- `EnvironmentSatisfaction`
- `YearsSinceLastPromotion`
- `JobSatisfaction`
- `JobInvolvement`
- `DistanceFromHome`

Additional features supported by two methods include:

- `MaritalStatus`
- `EducationField`
- `RoleTenureRatio`
- `Department`
- `TotalWorkingYears`
- `Age`

See [`docs/explainability_summary.md`](docs/explainability_summary.md).

---

## Project Structure

```text
Employee_Attrition_Prediction/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── docs/
│   ├── baseline_diagnostics.md
│   ├── baseline_modeling.md
│   ├── coefficient_analysis.md
│   ├── data_dictionary.md
│   ├── eda_findings.md
│   ├── eda_protocol.md
│   ├── explainability_summary.md
│   ├── feature_engineering.md
│   ├── final_holdout_evaluation.md
│   ├── hyperparameter_tuning.md
│   ├── imbalance_experiments.md
│   ├── modeling_protocol.md
│   ├── preprocessing.md
│   ├── shap_global.md
│   ├── shap_local.md
│   └── threshold_calibration.md
│
├── src/
│   └── attrition/
│       ├── calibration.py
│       ├── coefficients.py
│       ├── config.py
│       ├── data.py
│       ├── diagnostics.py
│       ├── eda.py
│       ├── explainability_summary.py
│       ├── features.py
│       ├── final_evaluation.py
│       ├── imbalance.py
│       ├── modeling.py
│       ├── permutation.py
│       ├── preprocessing.py
│       ├── shap_global.py
│       ├── shap_local.py
│       ├── split.py
│       ├── tuning.py
│       ├── validation.py
│       └── visualization.py
│
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated reports and plots are stored under:

```text
reports/generated/
```

These files are ignored by Git because they are reproducible from source.

---

## CLI Commands

After installation, the project exposes the following commands:

| Command | Purpose |
| --- | --- |
| `attrition-validate` | validate dataset assumptions |
| `attrition-eda` | generate machine-readable EDA |
| `attrition-eda-charts` | generate EDA visualizations |
| `attrition-baselines` | evaluate baseline models |
| `attrition-compare` | compare baseline performance and stability |
| `attrition-imbalance` | run imbalance-handling experiments |
| `attrition-tune` | run hyperparameter tuning |
| `attrition-calibrate` | analyze calibration and decision thresholds |
| `attrition-final-eval` | reproduce the frozen final evaluation |
| `attrition-coefficients` | generate Logistic Regression coefficient analysis |
| `attrition-permutation` | generate cross-validated permutation importance |
| `attrition-shap-global` | generate global SHAP analysis |
| `attrition-shap-local` | explain an individual training row |
| `attrition-explainability-report` | generate the cross-method explainability report |

---

## Local Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

### 3. Validate the dataset

```powershell
attrition-validate
```

### 4. Run quality checks

```powershell
python -m pytest
python -m ruff check .
```

Current validated quality gate:

```text
102 tests passed
Ruff: All checks passed
```

---

## Technology Stack

- Python 3.13+
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- SHAP
- Matplotlib
- Pydantic
- pydantic-settings
- pytest
- Ruff
- Hatchling / `pyproject.toml`
- Git / GitHub

---

## Current Development Roadmap

### Implemented

- dataset validation and schema checks;
- deterministic stratified data splitting;
- reproducible EDA;
- business feature engineering;
- leakage-safe preprocessing;
- baseline model comparison;
- imbalance-handling experiments;
- hyperparameter tuning;
- calibration and threshold selection;
- final holdout evaluation;
- coefficient analysis;
- permutation importance;
- global SHAP;
- local SHAP;
- cross-method explainability reporting;
- automated tests and linting.

### Planned

- Streamlit workforce analytics dashboard;
- interactive employee attrition-risk form;
- local SHAP explanations in the UI;
- model methodology/performance page;
- application caching and validation;
- CI/CD;
- Docker packaging;
- public deployment;
- final documentation polish.

---

## Responsible Use

This project uses a benchmark HR dataset and is intended as a machine learning engineering and analytics demonstration.

Model outputs must not be interpreted as:

- causal explanations of employee behavior;
- automated employment decisions;
- evidence for hiring, firing, promotion, compensation, or disciplinary action.

Explainability methods describe how the trained model behaves on the available data. They do not establish why an employee would leave an organization.

---

## Contributors

- **Anjali Singh** — [@Anjalisingh127](https://github.com/Anjalisingh127)
- **Gauri Jakhmola** — [@gaurijakhmola](https://github.com/gaurijakhmola)