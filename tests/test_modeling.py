"""Tests for Stage 4.1 baseline modeling."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from attrition.data import load_dataset
from attrition.modeling import (
    BASELINE_MODEL_NAMES,
    METRIC_NAMES,
    build_baseline_estimators,
    build_model_pipeline,
    evaluate_baseline,
)
from attrition.split import create_stratified_split

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def split():
    return create_stratified_split(load_dataset(DATASET), random_seed=42)


def test_baseline_registry_contains_expected_models() -> None:
    estimators = build_baseline_estimators(random_seed=42)

    assert tuple(estimators) == BASELINE_MODEL_NAMES


def test_model_pipeline_keeps_preprocessing_inside_pipeline(split) -> None:
    pipeline = build_model_pipeline(split.X_train, DummyClassifier(strategy="prior"))

    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_stratified_cv_operates_only_on_training_partition(split) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    seen_validation_positions: set[int] = set()

    for train_positions, validation_positions in cv.split(split.X_train, split.y_train):
        assert set(train_positions).isdisjoint(validation_positions)
        seen_validation_positions.update(int(value) for value in validation_positions)

    assert seen_validation_positions == set(range(len(split.X_train)))
    assert set(split.X_train.index).isdisjoint(split.X_test.index)


def test_dummy_baseline_returns_all_required_metrics(split) -> None:
    result = evaluate_baseline(
        "dummy",
        DummyClassifier(strategy="prior", random_state=42),
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=42,
    )

    assert result.training_rows == 1176
    assert result.positive_rows == 190
    assert result.folds == 5
    assert set(result.metrics) == set(METRIC_NAMES)

    for summary in result.metrics.values():
        assert len(summary.folds) == 5
        assert np.isfinite(summary.mean)
        assert np.isfinite(summary.std)


def test_dummy_baseline_exposes_class_imbalance(split) -> None:
    result = evaluate_baseline(
        "dummy",
        DummyClassifier(strategy="prior", random_state=42),
        split.X_train,
        split.y_train,
        folds=5,
        random_seed=42,
    )

    assert result.metrics["recall"].mean == pytest.approx(0.0)
    assert result.metrics["precision"].mean == pytest.approx(0.0)
    assert result.metrics["roc_auc"].mean == pytest.approx(0.5)
    assert result.metrics["pr_auc"].mean == pytest.approx(190 / 1176, abs=0.01)
