"""Tests for Stage 6.2 threshold and calibration diagnostics."""

from pathlib import Path

import numpy as np
import pytest

from attrition.calibration import (
    CALIBRATION_METHODS,
    DEFAULT_THRESHOLD,
    THRESHOLDS,
    build_tuned_logistic,
    calibration_bins,
    out_of_fold_probabilities,
    select_threshold,
    summarize_calibration,
    threshold_metrics,
    threshold_sweep,
)
from attrition.data import load_dataset
from attrition.split import create_stratified_split

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def split():
    return create_stratified_split(load_dataset(DATASET), random_seed=42)


def test_tuned_logistic_uses_stage_6_1_parameters(split) -> None:
    pipeline = build_tuned_logistic(split.X_train)
    model = pipeline.named_steps["model"]

    assert pytest.approx(0.3) == model.C
    assert model.penalty == "l2"
    assert model.solver == "liblinear"


def test_threshold_grid_contains_default_threshold() -> None:
    assert DEFAULT_THRESHOLD in THRESHOLDS
    assert min(THRESHOLDS) == pytest.approx(0.10)
    assert max(THRESHOLDS) == pytest.approx(0.70)


def test_threshold_metrics_match_simple_example() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = threshold_metrics(y_true, probabilities, 0.5)

    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        threshold_metrics(np.array([0, 1]), np.array([0.2, 0.8]), 1.0)


def test_threshold_selection_maximizes_f1() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])
    results = threshold_sweep(y_true, probabilities, (0.3, 0.5, 0.7))

    selected = select_threshold(results)

    assert selected.threshold == pytest.approx(0.5)
    assert selected.f1 == pytest.approx(1.0)


def test_calibration_summary_reports_probability_quality() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])

    summary = summarize_calibration("uncalibrated", y_true, probabilities)

    assert summary.method == "uncalibrated"
    assert summary.brier_score < 0.05
    assert summary.roc_auc == pytest.approx(1.0)
    assert summary.pr_auc == pytest.approx(1.0)


def test_calibration_bins_have_expected_columns() -> None:
    frame = calibration_bins(
        np.array([0, 0, 1, 1, 0, 1]),
        np.array([0.05, 0.2, 0.8, 0.9, 0.3, 0.7]),
        bins=3,
    )

    assert frame.columns.tolist() == [
        "mean_predicted_probability",
        "observed_positive_rate",
    ]
    assert not frame.empty


def test_calibration_methods_are_explicit() -> None:
    assert CALIBRATION_METHODS == ("uncalibrated", "sigmoid", "isotonic")


def test_small_oof_prediction_covers_each_training_row(split) -> None:
    sample_X = split.X_train.iloc[:300]
    sample_y = split.y_train.loc[sample_X.index]
    estimator = build_tuned_logistic(sample_X)

    probabilities = out_of_fold_probabilities(
        estimator,
        sample_X,
        sample_y,
        folds=2,
        random_seed=42,
    )

    assert len(probabilities) == len(sample_X)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_holdout_indices_remain_disjoint(split) -> None:
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
