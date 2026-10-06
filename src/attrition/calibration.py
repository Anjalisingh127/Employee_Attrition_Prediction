"""Training-only threshold analysis and probability calibration diagnostics."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.modeling import build_model_pipeline
from attrition.split import create_stratified_split

DEFAULT_THRESHOLD = 0.50
THRESHOLDS = tuple(float(value) for value in np.round(np.arange(0.10, 0.71, 0.02), 2))
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
DEFAULT_OUTPUT_DIR = Path("reports/generated")


@dataclass(frozen=True)
class ThresholdMetrics:
    """Classification metrics at one probability threshold."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    predicted_positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationSummary:
    """Probability-quality metrics for one calibration method."""

    method: str
    brier_score: float
    log_loss: float
    roc_auc: float
    pr_auc: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_tuned_logistic(
    X_train: pd.DataFrame,
    *,
    random_seed: int = 42,
):
    """Build the Stage 6.1 winning model with frozen hyperparameters."""

    return build_model_pipeline(
        X_train,
        LogisticRegression(
            C=0.3,
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=random_seed,
        ),
    )


def _outer_cv(folds: int, random_seed: int) -> StratifiedKFold:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)


def out_of_fold_probabilities(
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate one held-out probability for every training row."""

    probabilities = cross_val_predict(
        estimator,
        X_train,
        y_train,
        cv=_outer_cv(folds, random_seed),
        method="predict_proba",
        n_jobs=-1,
    )
    return np.asarray(probabilities[:, 1], dtype=float)


def threshold_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Calculate classification metrics for one threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        predicted_positive_rate=float(predictions.mean()),
    )


def threshold_sweep(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[ThresholdMetrics]:
    """Evaluate an explicit grid of probability thresholds."""

    if not thresholds:
        raise ValueError("at least one threshold is required")
    return [
        threshold_metrics(y_true, probabilities, threshold)
        for threshold in thresholds
    ]


def select_threshold(results: list[ThresholdMetrics]) -> ThresholdMetrics:
    """Select the highest-F1 threshold, then prefer higher recall and precision."""

    if not results:
        raise ValueError("at least one threshold result is required")

    return max(
        results,
        key=lambda result: (
            result.f1,
            result.recall,
            result.precision,
            result.threshold,
        ),
    )


def _calibrated_estimator(
    base_estimator: object,
    method: str,
) -> object:
    if method == "uncalibrated":
        return base_estimator
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(f"unknown calibration method: {method}")
    return CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=3,
        n_jobs=1,
    )


def calibration_probabilities(
    base_estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate leakage-safe OOF probabilities for one calibration strategy."""

    estimator = _calibrated_estimator(base_estimator, method)
    return out_of_fold_probabilities(
        estimator,
        X_train,
        y_train,
        folds=folds,
        random_seed=random_seed,
    )


def summarize_calibration(
    method: str,
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> CalibrationSummary:
    """Summarize discrimination and probability quality."""

    return CalibrationSummary(
        method=method,
        brier_score=float(brier_score_loss(y_true, probabilities)),
        log_loss=float(log_loss(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
    )


def calibration_bins(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Return observed-vs-predicted calibration points."""

    if bins < 2:
        raise ValueError("bins must be at least 2")

    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_positive_rate": observed,
        }
    )


def analyze_threshold_and_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    folds: int = 5,
    random_seed: int = 42,
) -> dict[str, object]:
    """Run Stage 6.2 diagnostics using training-only OOF predictions."""

    base = build_tuned_logistic(X_train, random_seed=random_seed)
    probability_sets = {
        method: calibration_probabilities(
            base,
            X_train,
            y_train,
            method,
            folds=folds,
            random_seed=random_seed,
        )
        for method in CALIBRATION_METHODS
    }

    calibration = [
        summarize_calibration(method, y_train, probabilities)
        for method, probabilities in probability_sets.items()
    ]
    recommended_calibration = min(
        calibration,
        key=lambda result: (result.brier_score, result.log_loss),
    )

    selected_probabilities = probability_sets[recommended_calibration.method]
    sweep = threshold_sweep(y_train, selected_probabilities)
    selected_threshold = select_threshold(sweep)
    default_metrics = threshold_metrics(
        y_train,
        selected_probabilities,
        DEFAULT_THRESHOLD,
    )

    return {
        "model": {
            "name": "logistic_regression",
            "C": 0.3,
            "penalty": "l2",
            "solver": "liblinear",
        },
        "probability_source": "out-of-fold predictions on training partition only",
        "threshold_selection_policy": "maximize F1; ties prefer recall, precision, threshold",
        "default_threshold": default_metrics.to_dict(),
        "selected_threshold": selected_threshold.to_dict(),
        "recommended_calibration": recommended_calibration.method,
        "calibration": [result.to_dict() for result in calibration],
        "thresholds": [result.to_dict() for result in sweep],
        "probabilities": probability_sets,
    }


def save_analysis(
    analysis: dict[str, object],
    y_train: pd.Series,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist reproducible generated diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / "threshold_analysis.csv"
    calibration_path = output_dir / "calibration_summary.csv"
    curve_path = output_dir / "calibration_curve.csv"
    json_path = output_dir / "threshold_calibration.json"

    pd.DataFrame(analysis["thresholds"]).to_csv(threshold_path, index=False)
    pd.DataFrame(analysis["calibration"]).to_csv(calibration_path, index=False)

    recommended = str(analysis["recommended_calibration"])
    probabilities = analysis["probabilities"][recommended]
    calibration_bins(y_train, probabilities).to_csv(curve_path, index=False)

    serializable = {
        key: value
        for key, value in analysis.items()
        if key != "probabilities"
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "threshold_csv": str(threshold_path),
        "calibration_csv": str(calibration_path),
        "calibration_curve_csv": str(curve_path),
        "summary_json": str(json_path),
    }


def main() -> None:
    """Run Stage 6.2 without evaluating the final holdout."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )
    analysis = analyze_threshold_and_calibration(
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=settings.random_seed,
    )
    artifacts = save_analysis(analysis, split.y_train)

    payload = {
        key: value
        for key, value in analysis.items()
        if key not in {"probabilities", "thresholds"}
    }
    payload["artifacts"] = artifacts
    payload["final_holdout_rows"] = len(split.X_test)
    payload["holdout_evaluated"] = False
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
