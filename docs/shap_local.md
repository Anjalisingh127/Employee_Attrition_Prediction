# Stage 7.4 — SHAP local explainability

Stage 7.4 adds individual employee-level SHAP explanations for the frozen model policy.

This is an explanation layer only. It does not reopen model selection or reuse the consumed final holdout.

## What the local report contains

For one employee selected from the 1,176-row training partition, the report includes:

- the frozen Logistic Regression probability;
- the frozen isotonic-calibrated probability;
- the fixed Stage 6 threshold (`0.38`);
- the resulting attrition/non-attrition decision;
- the strongest SHAP contributions pushing the Logistic Regression score upward;
- the strongest SHAP contributions pushing the score downward;
- the SHAP base value and additivity check;
- the original input values;
- a waterfall plot.

The calibrated probability and the SHAP explanation are deliberately reported as separate layers.

## Why SHAP does not decompose the calibrated probability

Stage 6 selected isotonic calibration around the tuned Logistic Regression. Isotonic calibration is a nonlinear mapping applied after the base model score.

Stage 7.4 therefore uses SHAP to explain the underlying Logistic Regression decision function and separately reports the calibrated final probability used by the frozen decision policy.

A local SHAP value:

- greater than zero pushes this employee's attrition score upward;
- less than zero pushes the score downward.

These contributions explain model behavior and are not causal statements about employee attrition.

## Additivity check

For a linear SHAP explanation:

```text
base value + sum(local SHAP values) ≈ Logistic Regression decision_function
```

The generated JSON records the reconstruction error so that the explanation can be checked programmatically.

## Selecting an employee

The CLI uses a zero-based position within the deterministic training partition.

Default example:

```powershell
attrition-shap-local
```

Select another training employee:

```powershell
attrition-shap-local --row-position 25
```

Change the number of displayed increasing/reducing contributions:

```powershell
attrition-shap-local --row-position 25 --top-n 8
```

This row position is an analysis locator, not an employee identifier.

## Generated artifacts

For row position 25, the command writes:

```text
reports/generated/shap_local_row_25.csv
reports/generated/shap_local_row_25.json
reports/generated/shap_local_row_25_waterfall.png
```

Generated artifacts remain ignored by Git.

## Training-row probability warning

The selected employee comes from the training partition because the final holdout has already been consumed and is not reused for explainability.

The reported probability is therefore a demonstration of the frozen prediction/explanation workflow, not a new generalization-performance measurement.

## Evaluation boundary

The 294-row final holdout is excluded from Stage 7.4.
