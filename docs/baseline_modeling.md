# Stage 4.1 — leakage-safe baseline modeling

Stage 4.1 establishes trustworthy training-only baselines before imbalance handling, hyperparameter tuning, or final holdout evaluation.

## Models

Three deliberately untuned baselines are compared:

- `DummyClassifier(strategy="prior")` — sanity check showing what class imbalance alone can achieve;
- `LogisticRegression` — interpretable linear baseline;
- `RandomForestClassifier` — nonlinear tree-ensemble baseline.

These are starting points, not final models.

## Cross-validation design

The already-reserved 294-row test partition remains untouched.

Only the 1,176-row training partition enters Stage 4.1. Model evaluation uses shuffled, deterministic **Stratified 5-Fold Cross-Validation** with random seed 42.

Each fold trains the complete pipeline:

```text
business feature engineering
        ↓
numeric scaling + categorical one-hot encoding
        ↓
classifier
```

Because preprocessing is inside the model pipeline, scaling statistics and categorical vocabularies are learned independently from each fold's training rows.

## Metrics

Each model reports fold-level values plus mean and population standard deviation for:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC / Average Precision

Accuracy is included for context but is not the primary selection criterion because only about 16% of records are positive attrition cases.

The dummy model is expected to demonstrate this limitation: it can achieve high accuracy by favoring the majority class while producing zero positive-class recall.

## Holdout policy

Stage 4.1 does **not** calculate any metric on the final test set. The holdout will be used only after model-selection, imbalance, and tuning decisions are fixed.
