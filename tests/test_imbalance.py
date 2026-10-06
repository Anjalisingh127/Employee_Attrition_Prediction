"""Tests for leakage-safe imbalance experiments."""

from pathlib import Path

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.linear_model import LogisticRegression

from attrition.data import load_dataset
from attrition.imbalance import (
    EXPERIMENT_NAMES,
    build_imbalance_experiments,
    evaluate_imbalance_experiment,
)
from attrition.split import create_stratified_split

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def split():
    return create_stratified_split(load_dataset(DATASET), random_seed=42)


def test_experiment_registry_contains_expected_strategies(split) -> None:
    experiments = build_imbalance_experiments(split.X_train)

    assert tuple(experiments) == EXPERIMENT_NAMES


def test_resampling_occurs_after_preprocessing_and_before_model(split) -> None:
    experiments = build_imbalance_experiments(split.X_train)

    ros = experiments["logistic_random_oversampling"]
    smote = experiments["logistic_smote"]

    assert list(ros.named_steps) == [
        "business_features",
        "columns",
        "sampler",
        "model",
    ]
    assert isinstance(ros.named_steps["sampler"], RandomOverSampler)
    assert isinstance(smote.named_steps["sampler"], SMOTE)


def test_class_weight_strategy_does_not_add_sampler(split) -> None:
    experiments = build_imbalance_experiments(split.X_train)
    pipeline = experiments["logistic_class_weight"]

    assert list(pipeline.named_steps) == ["business_features", "columns", "model"]
    assert pipeline.named_steps["model"].class_weight == "balanced"


def test_smote_pipeline_fits_only_training_rows(split) -> None:
    experiment = build_imbalance_experiments(split.X_train)["logistic_smote"]

    result = evaluate_imbalance_experiment(
        "logistic_smote",
        experiment,
        split.X_train,
        split.y_train,
        folds=3,
        random_seed=42,
    )

    assert result.training_rows == 1176
    assert result.positive_rows == 190
    assert result.folds == 3
    assert set(result.metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }
    assert all(np.isfinite(metric.mean) for metric in result.metrics.values())


def test_holdout_indices_are_disjoint_before_imbalance_experiments(split) -> None:
    assert set(split.X_train.index).isdisjoint(split.X_test.index)


def test_invalid_fold_count_is_rejected(split) -> None:
    experiment = build_imbalance_experiments(split.X_train)["logistic_class_weight"]

    with pytest.raises(ValueError, match="folds must be at least 2"):
        evaluate_imbalance_experiment(
            "logistic_class_weight",
            experiment,
            split.X_train,
            split.y_train,
            folds=1,
        )


def test_logistic_class_weight_configuration(split) -> None:
    experiment = build_imbalance_experiments(split.X_train)["logistic_class_weight"]
    model = experiment.named_steps["model"]

    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
