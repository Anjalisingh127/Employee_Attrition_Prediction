# Stage 5 — class-imbalance experiments

Stage 5 tests whether the minority attrition class can be handled more effectively than the untreated Stage 4 baselines.

## Strategies

Both Logistic Regression and Random Forest are evaluated with:

1. `class_weight="balanced"`
2. `RandomOverSampler`
3. `SMOTE`

The experiment therefore contains six model/imbalance combinations.

## Leakage control

Resampling must never occur before the cross-validation split.

For RandomOverSampler and SMOTE, the imbalanced-learn pipeline is:

```text
business feature engineering
        ↓
numeric scaling + one-hot encoding
        ↓
resampling on the current training fold only
        ↓
classifier
```

During each CV iteration, the sampler sees only the training portion of that fold. Validation rows are transformed but never resampled.

The 294-row final holdout remains completely outside Stage 5.

## Fair comparison

Stage 5 preserves the Stage 4 evaluation settings:

- 1,176-row training partition
- shuffled Stratified 5-Fold CV
- random seed 42
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC

No hyperparameter search or threshold tuning occurs here. The purpose is to isolate the effect of imbalance handling.

## Interpretation

Higher recall alone does not automatically make a strategy better. Oversampling can increase positive predictions while reducing precision or ranking quality.

The next decision should therefore compare each strategy against the untreated Stage 4 baseline using PR-AUC, recall, F1, precision, and fold stability together.
