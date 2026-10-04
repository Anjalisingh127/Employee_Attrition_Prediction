# Modeling protocol

This document defines the evaluation rules before any model is trained.

## Final holdout

The reference dataset is split once with:

- test size: 20%;
- random seed: 42;
- stratification on the binary attrition target.

For the 1,470-row reference dataset this produces 1,176 training rows and 294 test rows. The expected positive-class counts are 190 in training and 47 in test.

The test partition is the final holdout. Model selection, preprocessing decisions, resampling strategy, hyperparameter tuning, threshold exploration, and cross-validation must use training data only.

## Feature contract

The model target is `Attrition`, mapped as:

- `No -> 0`
- `Yes -> 1`

The following columns are excluded before modeling:

- `EmployeeNumber` — identifier, not a behavioral predictor;
- `EmployeeCount` — invariant;
- `Over18` — invariant;
- `StandardHours` — invariant.

This leaves 30 candidate predictor columns.

## Leakage controls

Future categorical encoding, scaling, imputation if ever required, feature selection, and resampling must live inside training/CV pipelines. SMOTE or other resampling must never be applied to the complete dataset or to the final test partition.

The final test partition should be evaluated only after the model-selection procedure has been fixed.
