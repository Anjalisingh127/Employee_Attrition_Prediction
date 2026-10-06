"""One-time final holdout evaluation for the frozen attrition model."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from attrition.calibration import build_tuned_logistic
from attrition.config import get_settings
from attrition.data import load_dataset, validate_dataset
from attrition.split import create_stratified_split

FINAL_THRESHOLD = 0.38
FINAL_CALIBRATION = "isotonic"
CALIBRATION_CV = 3
DEFAULT_OUTPUT_DIR = Path("reports/generated")


@dataclass(frozen=True)
class FinalMetrics:
    """Final holdout metrics for the frozen decision policy."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    predicted_positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ConfusionMatrixSummary:
    """Binary confusion-matrix counts."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_final_estimator(
    X_train: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> CalibratedClassifierCV:
    """Build the frozen Stage 6.3 estimator without fitting it."""

    base = build_tuned_logistic(X_train, random_seed=random_seed)
    return CalibratedClassifierCV(
        estimator=base,
        method=FINAL_CALIBRATION,
        cv=CALIBRATION_CV,
        n_jobs=1,
    )


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = FINAL_THRESHOLD,
) -> tuple[FinalMetrics, ConfusionMatrixSummary]:
    """Evaluate frozen probabilities at the frozen decision threshold."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    metrics = FinalMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        log_loss=float(log_loss(y_true, probabilities)),
        predicted_positive_rate=float(predictions.mean()),
    )
    matrix = ConfusionMatrixSummary(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
    return metrics, matrix


def evaluate_holdout_once(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_seed: int = 42,
) -> dict[str, object]:
    """Fit on training data and evaluate the frozen policy on the holdout."""

    estimator = build_final_estimator(X_train, random_seed=random_seed)
    estimator.fit(X_train, y_train)
    probabilities = np.asarray(estimator.predict_proba(X_test)[:, 1], dtype=float)

    metrics, matrix = evaluate_probabilities(
        y_test,
        probabilities,
        threshold=FINAL_THRESHOLD,
    )
    return {
        "model": {
            "name": "logistic_regression",
            "C": 0.3,
            "penalty": "l2",
            "solver": "liblinear",
        },
        "calibration": {
            "method": FINAL_CALIBRATION,
            "cv": CALIBRATION_CV,
        },
        "threshold": FINAL_THRESHOLD,
        "training_rows": len(X_train),
        "holdout_rows": len(X_test),
        "holdout_positive_rows": int(y_test.sum()),
        "metrics": metrics.to_dict(),
        "confusion_matrix": matrix.to_dict(),
        "holdout_evaluated": True,
        "holdout_consumed_for_final_evaluation": True,
        "post_holdout_policy": (
            "Do not change model, calibration, threshold, or preprocessing in response "
            "to these holdout metrics."
        ),
    }


def save_final_report(
    result: dict[str, object],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Persist the final evaluation report and confusion matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_holdout_evaluation.json"
    confusion_path = output_dir / "final_confusion_matrix.csv"

    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame([result["confusion_matrix"]]).to_csv(confusion_path, index=False)

    return {
        "summary_json": str(json_path),
        "confusion_matrix_csv": str(confusion_path),
    }


def main() -> None:
    """Execute the one-time final holdout evaluation."""

    settings = get_settings()
    df = load_dataset(Path(settings.dataset_path))
    validate_dataset(df)
    split = create_stratified_split(
        df,
        test_size=settings.test_size,
        random_seed=settings.random_seed,
    )

    result = evaluate_holdout_once(
        split.X_train,
        split.y_train,
        split.X_test,
        split.y_test,
        random_seed=settings.random_seed,
    )
    result["artifacts"] = save_final_report(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
