# Stage 6.3 — final holdout evaluation

Stage 6.3 performs the first intentional evaluation of the reserved 294-row holdout.

Everything used for this evaluation is frozen from earlier stages.

## Frozen model policy

The final evaluation uses:

- model: Logistic Regression
- `C=0.3`
- penalty: `l2`
- solver: `liblinear`
- calibration: isotonic
- calibration CV: 3 folds on training data
- decision threshold: `0.38`

No model, calibration, preprocessing, or threshold choice is learned from the holdout.

## Training and calibration

The final estimator is fitted using all 1,176 training rows.

`CalibratedClassifierCV(method="isotonic", cv=3)` performs calibration using training data only. The 294-row holdout is provided only after the frozen estimator has been fitted.

## Final metrics

The holdout report includes:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC / Average Precision
- Brier score
- log loss
- predicted-positive rate
- true negatives
- false positives
- false negatives
- true positives

## One-time evaluation policy

Run:

```powershell
attrition-final-eval
```

This command intentionally consumes the reserved holdout for final evaluation.

After the result is observed, do **not** alter model hyperparameters, calibration method, decision threshold, feature engineering, preprocessing, or imbalance handling in response to the holdout metrics. Doing so would turn the holdout into another validation set.

Any future material model changes require a new untouched evaluation dataset or an explicitly documented new evaluation protocol.

## Generated artifacts

The command writes:

```text
reports/generated/final_holdout_evaluation.json
reports/generated/final_confusion_matrix.csv
```

These generated artifacts remain ignored by Git. Verified final metrics can be summarized later in committed documentation after the run is validated.
