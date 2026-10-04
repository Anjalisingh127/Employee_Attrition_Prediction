# Employee Attrition Prediction & HR Analytics Platform

An end-to-end machine learning project for analyzing employee attrition patterns and estimating individual attrition risk using the IBM HR Analytics Employee Attrition dataset.

## Project status

This repository is being rebuilt from an earlier academic prototype into a reproducible, leakage-safe portfolio project. The current milestone establishes a clean project baseline and preserves the original public dataset for the next modeling stages.

## Contributors

- Anjali Singh — [@Anjalisingh127](https://github.com/Anjalisingh127)
- Gauri Jakhmola — [@gaurijakhmola](https://github.com/gaurijakhmola)

## Dataset

The project uses the public IBM HR Analytics Employee Attrition dataset with 1,470 employee records and 35 columns. It is a benchmark/synthetic dataset and should not be interpreted as production HR data.

Source data is stored in:

```text
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## Target architecture

The finished project will include:

1. reproducible data validation and preprocessing;
2. exploratory attrition analysis;
3. business-motivated feature engineering;
4. leakage-safe class-imbalance handling;
5. model comparison across Logistic Regression, Random Forest, XGBoost, and LightGBM;
6. stratified cross-validation and hyperparameter tuning;
7. evaluation with precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices;
8. probability threshold analysis and false-negative review;
9. SHAP-based global and employee-level explainability;
10. an interactive Streamlit HR analytics and risk-prediction dashboard.

## Important evaluation principle

Accuracy alone is not sufficient for this problem. Because the positive attrition class is imbalanced, the project will prioritize recall, F1, ROC-AUC, PR-AUC, calibration, and interpretability. Any final resume metrics will come from the rebuilt leakage-safe pipeline rather than the legacy prototype.

## Repository roadmap

- **Stage 1 — Data foundation:** dataset audit, schema validation, data dictionary, train/test strategy
- **Stage 2 — EDA:** attrition distribution and workforce risk patterns
- **Stage 3 — Feature engineering:** validated, non-leaking business features
- **Stage 4 — Modeling:** reproducible preprocessing + model baselines
- **Stage 5 — Imbalance experiments:** class weights, oversampling, SMOTE
- **Stage 6 — Tuning & evaluation:** stratified 5-fold CV and final test metrics
- **Stage 7 — Explainability:** feature importance, permutation importance, SHAP
- **Stage 8 — Streamlit application:** workforce analytics and employee risk scoring
- **Stage 9 — Engineering polish:** tests, CI, Docker, documentation, deployment

## Notes

The earlier repository contained notebooks, generated artifacts, experimental model files, and unrelated email-scanning code from an unfinished prototype. Those files were intentionally removed from the current branch so the project can be rebuilt with a clear, auditable implementation. Historical commits remain available in Git history.

No legacy performance claim should be treated as a final result until it has been reproduced using the new leakage-safe evaluation pipeline.
