# Stage 7.5 — explainability summary and recruiter-facing report

Stage 7.5 consolidates the completed explainability work into one reproducible, recruiter-readable summary.

## Why a synthesis layer is needed

No single explanation method is treated as ground truth.

The project now has:

- Logistic Regression coefficient analysis;
- cross-validated permutation importance;
- global SHAP;
- local SHAP.

Each method answers a different question. Stage 7.5 therefore emphasizes cross-method agreement instead of inventing a weighted importance score.

## Consensus rule

The summary maps transformed coefficient and SHAP terms back to their original business feature families where possible.

For example:

- `OverTime_Yes` and `OverTime_No` map to `OverTime`;
- all one-hot `JobRole_...` terms map to `JobRole`.

Each global method is ranked independently.

A feature is highlighted as a cross-method signal when it appears in the top 15 for at least two of:

1. absolute Logistic Regression coefficient magnitude;
2. cross-validated permutation importance by Average Precision decrease;
3. mean absolute SHAP value.

There is no arbitrary weighted composite score.

## Direction is reported separately

Feature-family consensus measures model reliance, not direction.

Direction is kept at the transformed-term level because categorical levels can point in opposite directions. For example, `OverTime_Yes` and `OverTime_No` should not be collapsed into one signed effect.

## Recruiter-facing output

Run:

```powershell
attrition-explainability-report
```

The command regenerates the global explainability analyses from the training partition and writes:

```text
reports/generated/explainability_consensus.csv
reports/generated/explainability_summary.json
reports/generated/explainability_recruiter_report.md
```

The Markdown report is intentionally concise enough for a reviewer to understand:

- which features are supported across multiple methods;
- which transformed terms push modeled attrition log-odds up or down;
- how permutation importance was validated;
- how global and local explanations fit together;
- the major interpretation guardrails.

## Guardrails

All Stage 7 findings explain model behavior rather than causal employee behavior.

Correlated and redundant variables can divide importance. One-hot categorical terms should be interpreted as encoded model contributions. SHAP and coefficient explanations describe the underlying Logistic Regression score, while isotonic calibration remains a separate probability mapping.

The consumed 294-row holdout is not reused.

## Completion criterion

Stage 7 is complete when this report passes the automated quality gate and regenerates successfully from the frozen training workflow.
