# Stage 7.3 — SHAP global explainability

Stage 7.3 adds global SHAP explanations for the frozen tuned Logistic Regression.

This stage explains the model; it does not reopen model selection.

## Scope

The explained estimator is the frozen underlying Logistic Regression:

- `C=0.3`
- L2 penalty
- `liblinear`
- Stage 3 business feature engineering
- numeric standardization
- categorical one-hot encoding

The model is fitted on the 1,176-row training partition only.

The final policy also uses isotonic calibration. That calibration layer is a nonlinear probability post-processing step and is not represented by the Logistic Regression SHAP values in this stage.

## Explainer

The implementation uses `shap.LinearExplainer`, which is designed for linear models.

The fitted preprocessing pipeline first converts the 30 original inputs into the same transformed feature matrix used by Logistic Regression. SHAP values are then calculated for all training rows and aligned with the transformed feature names.

## Global metrics

For every transformed feature, the report stores:

- `mean_abs_shap`: average absolute contribution magnitude;
- `mean_signed_shap`: average signed contribution;
- `positive_share`: fraction of rows where the feature contribution pushes the model output upward;
- `negative_share`: fraction of rows where the contribution pushes the model output downward.

Global ranking uses mean absolute SHAP value.

A large mean absolute SHAP value means the model relies strongly on that transformed feature across the analyzed population. It does not imply causality.

## Plots

Run:

```powershell
attrition-shap-global
```

The command writes:

```text
reports/generated/shap_global_importance.csv
reports/generated/shap_global_summary.json
reports/generated/shap_global_bar.png
reports/generated/shap_global_beeswarm.png
```

The bar plot summarizes global magnitude. The beeswarm plot adds row-level direction and feature-value context across the population.

## Relationship to Stages 7.1 and 7.2

- Stage 7.1 coefficient analysis explains the signed linear weights.
- Stage 7.2 permutation importance measures validation-score dependence on original features through the full calibrated pipeline.
- Stage 7.3 SHAP shows how transformed feature values contribute across individual rows and then aggregates those contributions globally.

Agreement across these views is stronger evidence of model reliance than any single explanation method alone.

## Limitations

SHAP explains model behavior, not real-world causal effects.

Correlated variables, engineered ratios, and one-hot encoded categories may divide or redistribute attribution. SHAP values should therefore be described as model contributions rather than causal employee-retention drivers.

## Evaluation boundary

The previously consumed 294-row final holdout is not reused for SHAP analysis.
