# Stage 7.2 — permutation importance

Stage 7.2 adds a model-agnostic global importance view for the frozen final policy.

It does not reopen model selection and it does not use the consumed final holdout.

## Method

The analysis uses shuffled Stratified 5-Fold cross-validation on the 1,176-row training partition.

For each outer fold:

1. the frozen final estimator is fitted on the fold's training rows;
2. performance is measured on that fold's held-out validation rows;
3. each original input feature is shuffled repeatedly on the validation rows;
4. the decrease in validation score is recorded;
5. importances are aggregated across folds and repeats.

The final calibrated estimator is used, so the importance calculation reflects the complete frozen prediction policy rather than only the raw Logistic Regression coefficient vector.

## Scoring

The primary importance score is **Average Precision**.

The secondary score is **ROC-AUC**.

Average Precision is primary because attrition is the minority class and later model development used minority-class-sensitive ranking metrics. Retaining ROC-AUC provides a second discrimination view and makes metric-dependent changes in feature ranking visible.

A positive importance means shuffling that feature reduced validation performance on average. Values near zero imply little measurable reliance under this evaluation. Negative values are retained rather than clipped because random shuffling can occasionally improve a validation score by chance.

## Original features rather than one-hot columns

Permutation occurs on the 30 original model inputs before feature engineering, scaling, and one-hot encoding.

This is useful alongside Stage 7.1:

- coefficient analysis explains individual transformed terms in the linear decision function;
- permutation importance measures dependence on an original business feature through the complete frozen pipeline.

## Limitations

Permutation importance is predictive, not causal.

Correlated or redundant features can share information. Shuffling one of them may cause only a small performance drop because another feature can preserve similar signal. Therefore low permutation importance does not prove that a variable is irrelevant in the real world.

## Run

```powershell
attrition-permutation
```

The command writes:

```text
reports/generated/permutation_importance.csv
reports/generated/permutation_importance.json
```

Generated artifacts remain ignored by Git.

## Evaluation boundary

The 294-row final holdout is not reused. All reported importances are estimated from held-out folds within the training partition and are for explanation only.
