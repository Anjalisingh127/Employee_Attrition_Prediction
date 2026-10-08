# Employee Attrition Prediction & HR Analytics Platform

A reproducible machine learning project for analyzing employee attrition patterns, estimating attrition risk, and explaining model behavior using the IBM HR Analytics Employee Attrition dataset.

The project was rebuilt from an earlier academic prototype into a leakage-safe, testable ML workflow with deterministic data splitting, cross-validation, imbalance experiments, hyperparameter tuning, probability calibration, threshold selection, final holdout evaluation, and multi-method explainability.

## Project status

**Completed through Stage 7.5 — Explainability Summary & Recruiter-Facing Report**

Current validated state:

- 1,470 validated employee records
- 30 original model inputs after removing target, identifier, and invariant fields
- deterministic 80/20 stratified split
- 1,176 training rows
- 294-row final holdout
- frozen final Logistic Regression policy
- isotonic probability calibration
- decision threshold: **0.38**
- final holdout evaluated once and then frozen
- coefficient, permutation, global SHAP, and local SHAP explainability
- cross-method recruiter-facing explanation summary
- **102 automated tests passing**
- Ruff static checks passing

The remaining project work is the Streamlit application and final engineering/deployment polish.

## Contributors

- Anjali Singh — [@Anjalisingh127](https://github.com/Anjalisingh127)
- Gauri Jakhmola — [@gaurijakhmola](https://github.com/gaurijakhmola)

## Final model

The final model policy was selected entirely from the training partition before the holdout was evaluated.

| Component | Frozen configuration |
| --- | --- |
| Model | Logistic Regression |
| Regularization | L2 |
| C | 0.3 |
| Solver | liblinear |
| Probability calibration | Isotonic |
| Calibration CV | 3 folds |
| Decision threshold | 0.38 |

The 294-row final holdout was evaluated only after model choice, calibration method, and threshold had been frozen.

## Final holdout results

| Metric | Result |
| --- | ---: |
| Accuracy | **87.07%** |
| Precision | **62.16%** |
| Recall | **48.94%** |
| F1 | **54.76%** |
| ROC-AUC | **81.08%** |
| Average Precision | **58.39%** |
| Brier score | **0.0967** |
| Log loss | **0.4393** |

Confusion matrix:

|  | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| Actual No | **233** | **14** |
| Actual Yes | **24** | **23** |

> Some earlier project outputs use the key name `pr_auc` while calling
> `average_precision_score`. In this README, that value is described precisely as
> **Average Precision (AP)**.

These results are treated as the final generalization estimate. No later explainability work changes the model, preprocessing, calibration method, or threshold in response to holdout performance.

## Dataset

The project uses the IBM HR Analytics Employee Attrition dataset:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Validated dataset contract:

- 1,470 rows
- 35 source columns
- target: `Attrition`
- 237 attrition cases
- 1,233 non-attrition cases
- attrition rate: approximately 16.12%
- no missing values
- no duplicate rows
- `EmployeeNumber` unique
- `EmployeeCount`, `Over18`, and `StandardHours` invariant

For modeling, the target, employee identifier, and invariant columns are excluded before learning begins.

See `docs/data_dictionary.md` for feature treatment details.

## Leakage-safe evaluation design

The project creates the final holdout before any learned transformation.

```text
Validated dataset
      |
      v
Stratified 80/20 split
      |
      +------------------------------+
      |                              |
      v                              v
Training partition              Final holdout
1,176 rows                      294 rows
      |                              |
      | CV / tuning / calibration    |
      | threshold selection          |
      | explainability               |
      v                              |
Frozen model policy                  |
      |                              |
      +----------------------------->|
                                     v
                           One-time final evaluation
```

Training-only operations include:

- scaling
- one-hot encoding
- engineered features
- resampling
- hyperparameter search
- calibration selection
- threshold selection
- explainability analysis

See `docs/modeling_protocol.md` for the evaluation rules.

## Feature engineering

Five deterministic business features are created without using the target:

- `IncomePerJobLevel`
- `CompanyTenureRatio`
- `RoleTenureRatio`
- `PromotionWaitRatio`
- `EarlyCareer`

Numeric features are standardized and categorical features are one-hot encoded inside the model pipeline so transformations are learned independently within each cross-validation fold.

See:

- `docs/feature_engineering.md`
- `docs/preprocessing.md`

## Modeling progression

### Baselines

The first benchmark compared:

- class-prior Dummy Classifier
- Logistic Regression
- Random Forest

All models were evaluated with shuffled Stratified 5-Fold Cross-Validation and full preprocessing inside each fold.

The untreated Logistic Regression produced the strongest overall baseline balance and became the main candidate for later tuning.

### Imbalance experiments

Controlled experiments evaluated:

- class weighting
- RandomOverSampler
- SMOTE

for Logistic Regression and Random Forest.

Resampling occurs only inside training folds through an imbalanced-learn pipeline.

The experiments showed the expected trade-off: class weighting improved recall substantially but reduced precision and Average Precision compared with the untreated Logistic Regression.

See `docs/imbalance_experiments.md`.

### Hyperparameter tuning

Three candidate strategies entered a two-stage search:

1. Logistic Regression
2. class-weighted Logistic Regression
3. Random Forest + SMOTE

The search used:

- bounded `RandomizedSearchCV`
- targeted `GridSearchCV`
- shuffled Stratified 5-Fold CV
- training data only

The selected model was:

```text
Logistic Regression
C = 0.3
penalty = l2
solver = liblinear
```

See `docs/hyperparameter_tuning.md`.

## Calibration and threshold selection

Stage 6.2 compared:

- uncalibrated probabilities
- sigmoid calibration
- isotonic calibration

using out-of-fold training predictions.

Isotonic calibration produced the best calibration quality under the predefined Brier-score-first selection policy.

The decision threshold was selected from training-only out-of-fold probabilities by maximizing F1 without inventing business cost weights.

Selected threshold:

```text
0.38
```

At the training-only selection stage, the threshold improved the recall/F1 trade-off relative to the default 0.50 threshold.

See `docs/threshold_calibration.md`.

## Explainability

The final model is explained through multiple complementary methods rather than one importance chart.

### 1. Logistic Regression coefficients

The transformed feature coefficients provide signed model direction and odds-ratio interpretation.

Examples of stronger higher modeled attrition log-odds terms:

- `JobRole_Laboratory Technician`
- `OverTime_Yes`
- `BusinessTravel_Travel_Frequently`
- `NumCompaniesWorked`
- `YearsSinceLastPromotion`

Examples of stronger lower modeled attrition log-odds terms:

- `OverTime_No`
- `BusinessTravel_Non-Travel`
- `Department_Research & Development`
- `JobRole_Research Director`
- `EducationField_Other`

These are model associations, not causal HR findings.

See `docs/coefficient_analysis.md`.

### 2. Cross-validated permutation importance

Permutation importance measures how much validation performance decreases when an original business feature is shuffled while the fitted estimator remains unchanged.

The frozen calibrated policy achieved:

- mean Average Precision: **0.6695 ± 0.0527**
- mean ROC-AUC: **0.8412 ± 0.0276**

across held-out folds of the training partition.

Among the strongest original-feature dependencies were:

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

See `docs/permutation_importance.md`.

### 3. Global SHAP

`shap.LinearExplainer` explains the frozen underlying Logistic Regression decision function using the complete training background.

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

Generated outputs include global bar and beeswarm plots.

See `docs/shap_global.md`.

### 4. Local SHAP

Individual training-partition predictions can be decomposed into risk-increasing and risk-reducing contributions.

The local workflow reports:

- base Logistic Regression probability
- isotonic-calibrated probability
- frozen threshold decision
- strongest positive SHAP contributions
- strongest negative SHAP contributions
- SHAP waterfall plot
- numerical additivity check

SHAP explains the underlying Logistic Regression score; it does not directly decompose the nonlinear isotonic calibration mapping.

See `docs/shap_local.md`.

### 5. Cross-method consensus

Stage 7.5 maps transformed terms back to shared feature families and compares rankings across:

- coefficient magnitude
- permutation importance
- global SHAP

No weighted composite score is invented.

Features appearing in the top tier of all three methods include:

- `OverTime`
- `JobRole`
- `BusinessTravel`
- `NumCompaniesWorked`
- `EnvironmentSatisfaction`
- `YearsSinceLastPromotion`
- `JobSatisfaction`
- `JobInvolvement`
- `DistanceFromHome`

Additional features supported by two methods include `MaritalStatus`, `EducationField`, `RoleTenureRatio`, `Department`, `TotalWorkingYears`, and `Age`.

See `docs/explainability_summary.md`.

## Project commands

After installation, the project exposes reproducible command-line entry points:

| Command | Purpose |
| --- | --- |
| `attrition-validate` | Validate dataset contract |
| `attrition-eda` | Generate machine-readable EDA |
| `attrition-eda-charts` | Generate EDA visualizations |
| `attrition-baselines` | Evaluate baseline models |
| `attrition-compare` | Compare baseline stability and ranking |
| `attrition-imbalance` | Run imbalance-handling experiments |
| `attrition-tune` | Run two-stage hyperparameter tuning |
| `attrition-calibrate` | Analyze calibration and decision thresholds |
| `attrition-final-eval` | Reproduce the frozen final evaluation |
| `attrition-coefficients` | Generate coefficient analysis |
| `attrition-permutation` | Generate permutation importance |
| `attrition-shap-global` | Generate global SHAP analysis |
| `attrition-shap-local` | Explain an individual training row |
| `attrition-explainability-report` | Generate the Stage 7 consensus report |

## Local setup

Clone the repository and create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Validate the repository:

```powershell
python -m pytest
python -m ruff check .
```

Current validated quality gate:

```text
102 tests passed
Ruff: All checks passed
```

## Project structure

```text
Employee_Attrition_Prediction/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
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
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated reports and plots are written under `reports/generated/` and intentionally ignored by Git because they can be reproduced from source.

## Technology stack

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

## Completed roadmap

- [x] Stage 1 — data validation and leakage-safe train/holdout foundation
- [x] Stage 2 — reproducible EDA and visual analysis
- [x] Stage 3 — deterministic feature engineering and preprocessing
- [x] Stage 4 — baseline modeling and diagnostics
- [x] Stage 5 — imbalance-handling experiments
- [x] Stage 6 — hyperparameter tuning, calibration, thresholding, final holdout evaluation
- [x] Stage 7 — coefficient, permutation, global SHAP, local SHAP, explainability synthesis
- [ ] Stage 8 — Streamlit analytics and employee-risk application
- [ ] Stage 9 — CI/CD, Docker, deployment, final documentation polish

## Interpretation and responsible-use notes

This project uses a benchmark HR dataset and is intended as a machine learning engineering and analytics demonstration.

Model outputs should not be interpreted as:

- causal explanations of employee behavior;
- automated employment decisions;
- evidence for hiring, firing, promotion, compensation, or disciplinary action.

Explainability methods describe how the trained model behaves on the available data. They do not establish why an employee would leave an organization.

## Repository history

The repository originally contained an older academic prototype with downloaded notebooks, generated model files, experimental artifacts, and unrelated email-scanning functionality.

The current branch was deliberately rebuilt into a clean, reproducible project while preserving historical Git commits. Legacy performance claims were not reused unless reproduced under the rebuilt leakage-safe evaluation protocol.
