"""Tests for Stage 6.1 two-pass hyperparameter tuning."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from attrition.data import load_dataset
from attrition.split import create_stratified_split
from attrition.tuning import (
    LOGISTIC_C_VALUES,
    PRIMARY_METRIC,
    TUNING_CANDIDATES,
    build_tuning_candidates,
    randomized_search_spaces,
    run_randomized_search,
    targeted_grid,
)

DATASET = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")


@pytest.fixture(scope="module")
def split():
    return create_stratified_split(load_dataset(DATASET), random_seed=42)


def test_tuning_registry_contains_stage_6_candidates(split) -> None:
    candidates = build_tuning_candidates(split.X_train)

    assert tuple(candidates) == TUNING_CANDIDATES


def test_random_forest_smote_pipeline_keeps_sampler_inside_search_pipeline(split) -> None:
    candidate = build_tuning_candidates(split.X_train)["random_forest_smote"]

    assert list(candidate.named_steps) == [
        "business_features",
        "columns",
        "sampler",
        "model",
    ]
    assert candidate.named_steps["model"].n_jobs == 1


def test_logistic_search_space_is_valid_for_liblinear() -> None:
    spaces = randomized_search_spaces()

    assert spaces["logistic_baseline"]["model__penalty"] == ["l1", "l2"]
    assert spaces["logistic_class_weight"]["model__C"] == list(LOGISTIC_C_VALUES)


def test_targeted_logistic_grid_stays_near_randomized_winner() -> None:
    grid = targeted_grid(
        "logistic_baseline",
        {"model__C": 1.0, "model__penalty": "l2"},
    )

    assert grid["model__C"] == [0.3, 1.0, 3.0]
    assert grid["model__penalty"] == ["l2"]


def test_targeted_random_forest_grid_preserves_winning_structure() -> None:
    grid = targeted_grid(
        "random_forest_smote",
        {
            "sampler__k_neighbors": 5,
            "model__n_estimators": 500,
            "model__max_depth": 10,
            "model__min_samples_split": 5,
            "model__min_samples_leaf": 2,
            "model__max_features": "sqrt",
        },
    )

    assert grid["model__n_estimators"] == [400, 500, 600]
    assert grid["model__max_depth"] == [5, 10, 15]
    assert grid["sampler__k_neighbors"] == [5]


def test_unknown_candidate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tuning candidate"):
        targeted_grid("unknown", {})


def test_small_randomized_search_uses_pr_auc_and_training_rows_only(split) -> None:
    estimator = build_tuning_candidates(split.X_train)["logistic_baseline"]
    sample_X = split.X_train.iloc[:300]
    sample_y = split.y_train.loc[sample_X.index]

    search = run_randomized_search(
        "logistic_baseline",
        estimator,
        sample_X,
        sample_y,
        folds=2,
        random_seed=42,
        n_iter=2,
    )

    assert isinstance(search, RandomizedSearchCV)
    assert search.refit == PRIMARY_METRIC
    assert len(search.cv_results_["params"]) == 2
    assert isinstance(search.best_estimator_.named_steps["model"].C, float)


def test_search_types_are_exposed_by_sklearn() -> None:
    assert issubclass(RandomizedSearchCV, object)
    assert issubclass(GridSearchCV, object)


def test_holdout_indices_remain_disjoint(split) -> None:
    assert isinstance(split.X_train, pd.DataFrame)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
