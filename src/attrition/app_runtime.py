"""Shared, testable services for the Streamlit application."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from attrition.data import DatasetAudit, load_dataset, validate_dataset
from attrition.final_evaluation import (
    CALIBRATION_CV,
    FINAL_CALIBRATION,
    FINAL_THRESHOLD,
    build_final_estimator,
)
from attrition.split import DatasetSplit, create_stratified_split

FINAL_HOLDOUT_METRICS = {
    "accuracy": 0.8707482993197279,
    "precision": 0.6216216216216216,
    "recall": 0.48936170212765956,
    "f1": 0.5476190476190477,
    "roc_auc": 0.8107503155893537,
    "average_precision": 0.583943772411601,
    "brier_score": 0.09668895979409995,
    "log_loss": 0.43928700425726425,
}

FINAL_CONFUSION_MATRIX = {
    "true_negative": 233,
    "false_positive": 14,
    "false_negative": 24,
    "true_positive": 23,
}

FINAL_MODEL_POLICY = {
    "model": "Logistic Regression",
    "regularization": "L2",
    "C": 0.3,
    "solver": "liblinear",
    "calibration": FINAL_CALIBRATION,
    "calibration_cv": CALIBRATION_CV,
    "threshold": FINAL_THRESHOLD,
}


def load_validated_dataset(path: Path) -> tuple[pd.DataFrame, DatasetAudit]:
    """Load the source dataset and enforce the project data contract."""

    df = load_dataset(path)
    audit = validate_dataset(df)
    return df, audit


def build_training_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> DatasetSplit:
    """Recreate the deterministic training/holdout boundary used by the project."""

    return create_stratified_split(
        df,
        test_size=test_size,
        random_seed=random_seed,
    )


def fit_frozen_prediction_model(
    split: DatasetSplit,
    *,
    random_seed: int = 42,
):
    """Fit the frozen prediction policy on the training partition only."""

    estimator = build_final_estimator(
        split.X_train,
        random_seed=random_seed,
    )
    estimator.fit(split.X_train, split.y_train)
    return estimator


def overview_snapshot(audit: DatasetAudit) -> dict[str, object]:
    """Return concise dataset facts for the application overview."""

    return {
        "employees": audit.rows,
        "source_features": audit.columns,
        "attrition_cases": audit.attrition_yes,
        "non_attrition_cases": audit.attrition_no,
        "attrition_rate": audit.attrition_rate,
        "missing_values": audit.missing_values,
        "duplicate_rows": audit.duplicate_rows,
    }


def methodology_snapshot() -> dict[str, object]:
    """Return immutable model/evaluation facts without rescoring the holdout."""

    return {
        "model_policy": FINAL_MODEL_POLICY.copy(),
        "holdout_metrics": FINAL_HOLDOUT_METRICS.copy(),
        "confusion_matrix": FINAL_CONFUSION_MATRIX.copy(),
        "evaluation_policy": (
            "The final 294-row holdout was evaluated once after model, calibration, "
            "and threshold selection were frozen. The application does not rescore or "
            "retune against that holdout."
        ),
    }


def audit_as_dict(audit: DatasetAudit) -> dict[str, object]:
    """Convert a dataset audit to a cache-friendly plain dictionary."""

    return asdict(audit)
