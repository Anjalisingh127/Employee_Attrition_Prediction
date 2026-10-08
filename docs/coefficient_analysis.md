# Stage 7.1 — Logistic Regression coefficient analysis

Stage 7.1 adds global interpretability for the frozen Logistic Regression selected before final holdout evaluation.

This is an explanation stage, not another model-selection stage.

## What is explained

The final Stage 6 policy uses isotonic calibration around the tuned Logistic Regression. Isotonic calibration changes the probability mapping but does not expose one global coefficient vector.

For that reason, Stage 7.1 fits the already-frozen underlying Logistic Regression configuration on the full 1,176-row training partition and explains its decision function:

- `C=0.3`
- L2 penalty
- `liblinear` solver
- existing Stage 3 preprocessing and feature engineering

The holdout is excluded from coefficient fitting and interpretation.

## Feature names

The fitted preprocessing pipeline exposes the transformed feature names that enter Logistic Regression. Those names are aligned directly with `model.coef_`.

The full coefficient table contains:

- readable feature name;
- transformed pipeline feature name;
- coefficient;
- odds ratio (`exp(coefficient)`);
- absolute coefficient magnitude;
- direction.

## Interpretation

For the positive attrition class:

- a **positive coefficient** increases the model's attrition log-odds;
- a **negative coefficient** decreases the model's attrition log-odds;
- larger absolute magnitudes have stronger influence on the linear decision function.

Numeric inputs are standardized before modeling, so their coefficients reflect a one-standard-deviation change in the transformed numeric feature.

Categorical features are one-hot encoded. Their coefficients describe the model's encoded category contribution, but should not be presented as causal effects or as isolated real-world interventions.

Because correlated variables, regularization, engineered ratios, and one-hot categories can share predictive signal, coefficient magnitude is model-specific evidence rather than a causal ranking of HR factors.

## Run

```powershell
attrition-coefficients
```

The command prints the top positive and negative coefficient directions and writes:

```text
reports/generated/logistic_coefficients.csv
reports/generated/coefficient_analysis.json
```

Generated artifacts remain ignored by Git.

## Evaluation boundary

Stage 7.1 does not alter the frozen model, calibration method, threshold, preprocessing, or final Stage 6 metrics. The previously consumed 294-row holdout is not used to fit or rank coefficient explanations.
