"""Tests for Stage 6.3 final evaluation utilities.

These tests intentionally do not evaluate the real project holdout.
"""

import numpy as np
import pytest

from attrition.final_evaluation import (
    CALIBRATION_CV,
    FINAL_CALIBRATION,
    FINAL_THRESHOLD,
    evaluate_probabilities,
)


def test_final_policy_constants_are_frozen() -> None:
    assert pytest.approx(0.38) == FINAL_THRESHOLD
    assert FINAL_CALIBRATION == "isotonic"
    assert CALIBRATION_CV == 3


def test_final_metrics_from_synthetic_probabilities() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.25, 0.65, 0.90])

    metrics, matrix = evaluate_probabilities(y_true, probabilities)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert matrix.true_negative == 2
    assert matrix.false_positive == 0
    assert matrix.false_negative == 0
    assert matrix.true_positive == 2


def test_final_metrics_include_probability_quality() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    metrics, _ = evaluate_probabilities(y_true, probabilities)

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.brier_score < 0.05
    assert metrics.log_loss > 0.0


def test_invalid_final_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_probabilities(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            threshold=0.0,
        )
