"""Tests for Streamlit application services that do not require UI rendering."""

from attrition.app_runtime import (
    FINAL_CONFUSION_MATRIX,
    FINAL_HOLDOUT_METRICS,
    FINAL_MODEL_POLICY,
    methodology_snapshot,
    overview_snapshot,
)
from attrition.data import DatasetAudit


def _audit() -> DatasetAudit:
    return DatasetAudit(
        rows=1470,
        columns=35,
        missing_values=0,
        duplicate_rows=0,
        unique_employee_ids=1470,
        attrition_yes=237,
        attrition_no=1233,
        attrition_rate=237 / 1470,
        invariant_columns=("EmployeeCount", "Over18", "StandardHours"),
    )


def test_overview_snapshot_uses_validated_audit() -> None:
    snapshot = overview_snapshot(_audit())

    assert snapshot["employees"] == 1470
    assert snapshot["attrition_cases"] == 237
    assert snapshot["attrition_rate"] == 237 / 1470
    assert snapshot["missing_values"] == 0


def test_final_model_policy_is_frozen() -> None:
    assert FINAL_MODEL_POLICY["model"] == "Logistic Regression"
    assert FINAL_MODEL_POLICY["C"] == 0.3
    assert FINAL_MODEL_POLICY["calibration"] == "isotonic"
    assert FINAL_MODEL_POLICY["threshold"] == 0.38


def test_final_metrics_match_validated_holdout() -> None:
    assert FINAL_HOLDOUT_METRICS["accuracy"] > 0.87
    assert FINAL_HOLDOUT_METRICS["roc_auc"] > 0.81
    assert FINAL_HOLDOUT_METRICS["average_precision"] > 0.58


def test_confusion_matrix_counts_294_holdout_rows() -> None:
    assert sum(FINAL_CONFUSION_MATRIX.values()) == 294


def test_methodology_snapshot_returns_copies() -> None:
    first = methodology_snapshot()
    second = methodology_snapshot()

    first["model_policy"]["threshold"] = 0.99

    assert second["model_policy"]["threshold"] == 0.38
