# Stage 4.2 — baseline comparison and diagnostics

Stage 4.2 converts the Stage 4.1 cross-validation output into a transparent model-comparison layer. It does not introduce new estimators, tune hyperparameters, change thresholds, or evaluate the final holdout.

## Selection policy

Models are ranked lexicographically using:

1. **PR-AUC / Average Precision** — primary metric because attrition is the minority class;
2. **Recall** — first tie-breaker because missing potential attrition cases is operationally important;
3. **ROC-AUC** — second tie-breaker for overall ranking quality.

The project deliberately does **not** create a weighted composite score. Arbitrary weights can hide trade-offs between precision, recall, and ranking performance.

Accuracy, precision, and F1 remain visible in the comparison table but do not determine the primary ranking.

## Fold-stability diagnostics

For PR-AUC, recall, and ROC-AUC, the report preserves:

- cross-validation mean;
- standard deviation;
- minimum fold score;
- maximum fold score;
- observed fold range.

No arbitrary "stable/unstable" label is assigned. The raw spread is reported so later decisions can consider both average performance and consistency.

## Generated artifacts

Run:

```powershell
attrition-compare
```

This regenerates:

```text
reports/generated/baseline_comparison.csv
reports/generated/baseline_diagnostics.json
```

The generated files remain ignored by Git. The repository stores the code, tests, and methodology needed to reproduce them.

## Holdout policy

The command recreates the fixed Stage 1 train/test split and evaluates baselines only with five-fold cross-validation on the 1,176-row training partition.

The 294-row final test set remains untouched.

## Decision boundary

The Stage 4.2 recommendation identifies the baseline that should lead into the next experiment stage. It is **not** the final production-model choice. The recommendation can change after class-imbalance experiments, hyperparameter tuning, threshold analysis, calibration, and final holdout evaluation.
