# Stage 6.2 — threshold analysis and calibration

Stage 6.2 freezes the Stage 6.1 winning estimator configuration and studies two separate questions:

1. which decision threshold gives the best classification trade-off on training-only out-of-fold predictions;
2. whether probability calibration improves the reliability of predicted attrition probabilities.

The final 294-row test partition remains untouched.

## Frozen model

Stage 6.1 selected Logistic Regression with:

- `C=0.3`
- `penalty="l2"`
- `solver="liblinear"`

Stage 6.2 does not retune those hyperparameters.

## Out-of-fold probabilities

All threshold and calibration diagnostics use out-of-fold probabilities from shuffled Stratified 5-Fold CV.

Each training row receives a probability from a model that was not fitted on that row. Threshold selection therefore does not reuse in-sample fitted probabilities.

## Threshold policy

The analysis evaluates thresholds from 0.10 through 0.70 in increments of 0.02.

The selected threshold maximizes **F1**. If thresholds tie on F1, the rule prefers:

1. higher recall;
2. higher precision;
3. the higher threshold.

No financial or HR cost weighting is invented because the project does not have a validated business cost matrix.

The default 0.50 threshold is reported alongside the selected threshold.

## Calibration assessment

Three probability strategies are compared:

- uncalibrated tuned Logistic Regression;
- sigmoid calibration;
- isotonic calibration.

Sigmoid and isotonic calibration are fitted with `CalibratedClassifierCV(cv=3)` inside each outer cross-validation training fold. This creates a nested calibration process and keeps outer validation rows unavailable to the calibrator.

The calibration recommendation is the method with the lowest out-of-fold **Brier score**, with log loss as the tie-breaker.

The report also preserves ROC-AUC and PR-AUC so calibration improvements are not interpreted without checking discrimination.

## Generated artifacts

Run:

```powershell
attrition-calibrate
```

The command writes reproducible generated artifacts under `reports/generated/`:

- `threshold_analysis.csv`
- `calibration_summary.csv`
- `calibration_curve.csv`
- `threshold_calibration.json`

Generated files remain ignored by Git.

## Stage boundary

The selected threshold and calibration method are training-derived model-selection choices. Stage 6.3 should freeze these decisions before evaluating the final holdout exactly once.
