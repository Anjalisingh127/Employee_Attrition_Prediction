# Stage 6.1 — hyperparameter tuning

Stage 6.1 tunes only the candidates that survived the baseline and imbalance experiments:

1. untreated Logistic Regression;
2. Logistic Regression with `class_weight="balanced"`;
3. Random Forest with fold-local SMOTE.

The final 294-row holdout is still excluded from every search.

## Search objective

Both search passes optimize **PR-AUC / Average Precision** because attrition is the minority class and Stage 4 established PR-AUC as the primary model-ranking metric.

All other Stage 4 metrics remain available in each search result, but only PR-AUC determines `best_estimator_`.

## Two-pass strategy

### Pass 1 — RandomizedSearchCV

A bounded randomized search explores the wider parameter space using shuffled Stratified 5-Fold CV.

Logistic Regression explores regularization strength and L1/L2 penalty with the `liblinear` solver.

Random Forest + SMOTE explores:

- SMOTE neighbor count;
- tree count;
- maximum depth;
- minimum split size;
- minimum leaf size;
- maximum feature rule.

### Pass 2 — targeted GridSearchCV

The randomized-search winner defines a smaller local grid.

For Logistic Regression, the grid evaluates neighboring `C` values around the winner while keeping the winning penalty.

For Random Forest + SMOTE, the grid keeps most winning structural parameters fixed and refines tree count and maximum depth around the randomized winner.

This prevents an unnecessarily large exhaustive grid while still verifying the local parameter region.

## Leakage controls

Every candidate is a complete pipeline. Preprocessing, SMOTE where applicable, and the estimator are refitted separately inside each CV training fold.

The test partition is not supplied to `RandomizedSearchCV`, `GridSearchCV`, scoring, model selection, or parameter selection.

## Parallelism

The search object parallelizes CV jobs with `n_jobs=-1`. The Random Forest estimator itself uses one thread during tuning to avoid nested parallel execution.

## Run

```powershell
attrition-tune
```

The command prints the best randomized-search and targeted-grid PR-AUC plus best parameters for each candidate. These remain training-only model-selection results; they are not final test metrics.
